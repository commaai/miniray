#!/usr/bin/env python
"""Integration and load tests for miniray protocol V2.

These tests require Redis. They will attempt to use:
1. A local Redis on localhost:6379
2. A Docker Redis container (started automatically)

Skip with: pytest test_integration.py -k "not integration"
"""

import os
import time
import struct
import subprocess
import cloudpickle
import msgpack
import pytest
import redis

from protocol import (
    PROTOCOL_VERSION, MSG_TYPE_TASK, MSG_TYPE_RESULT_INLINE,
    MSG_TYPE_RESULT_INDIRECT, MSG_TYPE_RESULT_ERROR,
)


def is_redis_available(host="localhost", port=6379):
    """Check if Redis is available."""
    try:
        r = redis.StrictRedis(host=host, port=port, socket_connect_timeout=1)
        r.ping()
        return True
    except (redis.ConnectionError, redis.TimeoutError):
        return False


def start_redis_container():
    """Start a Redis container for testing, return port."""
    port = 16379  # Use non-standard port to avoid conflicts
    try:
        # Stop any existing test container
        subprocess.run(
            ["docker", "rm", "-f", "miniray-test-redis"],
            capture_output=True,
            timeout=10
        )
        # Start new container
        result = subprocess.run(
            ["docker", "run", "-d", "--name", "miniray-test-redis",
             "-p", f"{port}:6379", "redis:7-alpine"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            return None
        # Wait for Redis to be ready
        for _ in range(30):
            if is_redis_available("localhost", port):
                return port
            time.sleep(0.1)
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def stop_redis_container():
    """Stop the Redis test container."""
    try:
        subprocess.run(
            ["docker", "rm", "-f", "miniray-test-redis"],
            capture_output=True,
            timeout=10
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


@pytest.fixture(scope="module")
def redis_port():
    """Fixture that provides a Redis port, starting container if needed."""
    # Try local Redis first
    if is_redis_available("localhost", 6379):
        yield 6379
        return

    # Try to start Docker Redis
    port = start_redis_container()
    if port:
        yield port
        stop_redis_container()
        return

    pytest.skip("Redis not available (no local Redis or Docker)")


@pytest.fixture
def redis_client(redis_port):
    """Fixture that provides a Redis client and cleans up after test."""
    client = redis.StrictRedis(host="localhost", port=redis_port, db=15)
    client.flushdb()
    yield client
    client.flushdb()
    client.close()


class TestRedisProtocolIntegration:
    """Integration tests with real Redis."""

    def test_task_roundtrip_through_redis(self, redis_client):
        """Task should survive being pushed to and popped from Redis."""
        queue = "test-task-queue"

        # Pack a task
        pickled_fn = cloudpickle.dumps(lambda x: x * 2)
        pickled_args = cloudpickle.dumps(([(42,)], {}))
        payload = msgpack.packb({
            "uuid": "test-uuid",
            "job": queue,
            "function_ptr": "",
            "fn": pickled_fn,
            "args": pickled_args,
        }, use_bin_type=True)
        header = struct.pack(">IBB", len(payload) + 2, PROTOCOL_VERSION, MSG_TYPE_TASK)
        packed_task = header + payload

        # Push to Redis
        redis_client.lpush(queue, packed_task)

        # Pop from Redis
        raw_task = redis_client.rpop(queue)

        # Verify it's identical
        assert raw_task == packed_task

        # Verify we can unpack it
        length, version, msg_type = struct.unpack(">IBB", raw_task[:6])
        assert version == PROTOCOL_VERSION
        assert msg_type == MSG_TYPE_TASK

        unpacked = msgpack.unpackb(raw_task[6:], raw=False)
        assert unpacked["uuid"] == "test-uuid"
        assert unpacked["fn"] == pickled_fn

    def test_result_roundtrip_through_redis(self, redis_client):
        """Result should survive being pushed to and popped from Redis."""
        queue = "fq-test-result-queue"

        # Pack an inline result
        result_data = cloudpickle.dumps([{"succeeded": True, "result": 84}])
        result_msg = msgpack.packb({
            "job": "test-job",
            "worker": "test-worker",
            "task_uuid": "uuid-123",
            "result": result_data,
        }, use_bin_type=True)
        header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INLINE)
        packed_result = header + result_msg

        # Push to Redis
        redis_client.lpush(queue, packed_result)

        # Pop from Redis
        raw_result = redis_client.rpop(queue)

        assert raw_result == packed_result

    def test_blpop_receives_result(self, redis_client):
        """BLPOP should receive results pushed to queue."""
        queue = "fq-blpop-test"

        # Pack a result
        result_msg = msgpack.packb({
            "job": "test",
            "worker": "w1",
            "task_uuid": "uuid",
            "result": b"data",
        }, use_bin_type=True)
        header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INLINE)
        packed = header + result_msg

        # Push in a separate "thread" (we'll just push before blpop with timeout)
        redis_client.lpush(queue, packed)

        # BLPOP should return immediately
        result = redis_client.blpop(queue, timeout=1)

        assert result is not None
        key, value = result
        assert value == packed

    def test_blpop_timeout_returns_none(self, redis_client):
        """BLPOP should return None on timeout."""
        result = redis_client.blpop("nonexistent-queue", timeout=1)
        assert result is None

    def test_multiple_results_fifo_order(self, redis_client):
        """Results should be returned in FIFO order."""
        queue = "fq-fifo-test"

        # Push 3 results
        for i in range(3):
            result_msg = msgpack.packb({
                "job": "test",
                "worker": "w1",
                "task_uuid": f"uuid-{i}",
                "result": f"result-{i}".encode(),
            }, use_bin_type=True)
            header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INLINE)
            redis_client.lpush(queue, header + result_msg)

        # Pop in order (RPOP for FIFO since we LPUSH)
        for i in range(3):
            raw = redis_client.rpop(queue)
            payload = msgpack.unpackb(raw[6:], raw=False)
            assert payload["task_uuid"] == f"uuid-{i}"


class TestLoadPerformance:
    """Load tests to measure protocol performance."""

    def test_task_packing_throughput(self):
        """Measure task packing throughput."""
        from miniray.executor import Executor
        from unittest.mock import patch

        with patch('miniray.executor.redis.StrictRedis'):
            executor = object.__new__(Executor)
            executor.submit_queue_id = "bench-queue"

        pickled_fn = cloudpickle.dumps(lambda x: x)
        iterations = 10000

        start = time.perf_counter()
        for i in range(iterations):
            executor._pack_task("", pickled_fn, [(i,)], {}, f"uuid-{i}")
        elapsed = time.perf_counter() - start

        tasks_per_sec = iterations / elapsed
        print(f"\nTask packing: {tasks_per_sec:.0f} tasks/sec ({elapsed*1000/iterations:.3f} ms/task)")

        # Should be able to pack at least 10k tasks/sec
        assert tasks_per_sec > 10000, f"Task packing too slow: {tasks_per_sec:.0f}/sec"

    def test_result_unpacking_throughput(self):
        """Measure result unpacking throughput."""
        from miniray.executor import Executor, MiniraySubTaskResult
        from concurrent.futures import Future
        from unittest.mock import patch

        with patch('miniray.executor.redis.StrictRedis'):
            executor = object.__new__(Executor)
            executor._futures = {}
            executor._submit_redis_master = type('Mock', (), {'delete': lambda *a: None})()

        # Pre-generate messages
        iterations = 10000
        messages = []
        for i in range(iterations):
            task_uuid = f"uuid-{i}"
            executor._futures[task_uuid] = [Future()]

            subtasks = [MiniraySubTaskResult(True, "", "", i)]
            result_msg = msgpack.packb({
                "job": "test",
                "worker": "w1",
                "task_uuid": task_uuid,
                "result": cloudpickle.dumps(subtasks),
            }, use_bin_type=True)
            header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INLINE)
            messages.append(header + result_msg)

        start = time.perf_counter()
        for msg in messages:
            executor._unpack_result(msg)
        elapsed = time.perf_counter() - start

        results_per_sec = iterations / elapsed
        print(f"\nResult unpacking: {results_per_sec:.0f} results/sec ({elapsed*1000/iterations:.3f} ms/result)")

        # Should be able to unpack at least 5k results/sec
        assert results_per_sec > 5000, f"Result unpacking too slow: {results_per_sec:.0f}/sec"

    def test_redis_roundtrip_latency(self, redis_client):
        """Measure Redis push/pop latency."""
        queue = "bench-latency"
        iterations = 1000

        # Create a sample message
        result_msg = msgpack.packb({
            "job": "test",
            "worker": "w1",
            "task_uuid": "uuid",
            "result": b"x" * 100,
        }, use_bin_type=True)
        header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INLINE)
        message = header + result_msg

        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            redis_client.lpush(queue, message)
            redis_client.rpop(queue)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # Convert to ms

        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        print(f"\nRedis roundtrip: avg={avg_latency:.3f}ms, p50={p50:.3f}ms, p99={p99:.3f}ms")

        # Local Redis should have sub-millisecond latency
        assert avg_latency < 5, f"Redis latency too high: {avg_latency:.3f}ms"

    def test_inline_vs_indirect_size_comparison(self):
        """Compare message sizes for inline vs indirect results."""
        from miniray.executor import MiniraySubTaskResult

        # Small result (should be inline)
        small_result = MiniraySubTaskResult(True, "", "", {"key": "value"})
        small_data = cloudpickle.dumps([small_result])

        inline_msg = msgpack.packb({
            "job": "test-job",
            "worker": "worker-1",
            "task_uuid": "uuid-123",
            "result": small_data,
        }, use_bin_type=True)
        inline_size = len(inline_msg) + 6  # +6 for header

        indirect_msg = msgpack.packb({
            "job": "test-job",
            "worker": "worker-1",
            "task_uuid": "uuid-123",
            "host": "worker-redis-hostname",
            "key": "miniray-result-key-uuid",
        }, use_bin_type=True)
        indirect_size = len(indirect_msg) + 6

        print(f"\nSmall result ({len(small_data)} bytes payload):")
        print(f"  Inline message: {inline_size} bytes")
        print(f"  Indirect message: {indirect_size} bytes (+ separate payload)")
        print(f"  Inline saves: {indirect_size} bytes + 1 RTT")

        # Inline should include the payload but avoid the separate fetch
        assert inline_size > indirect_size  # Inline is larger but saves RTT


class TestProtocolVersioning:
    """Test protocol version handling."""

    def test_version_byte_position(self):
        """Version byte should be at position 4."""
        payload = msgpack.packb({"test": "data"})
        header = struct.pack(">IBB", len(payload) + 2, PROTOCOL_VERSION, MSG_TYPE_TASK)
        message = header + payload

        # Byte 4 (0-indexed) should be version
        assert message[4] == PROTOCOL_VERSION

    def test_different_versions_distinguishable(self):
        """Different protocol versions should be easily distinguishable."""
        payload = msgpack.packb({"test": "data"})

        v1_header = struct.pack(">IBB", len(payload) + 2, 0x01, MSG_TYPE_TASK)
        v2_header = struct.pack(">IBB", len(payload) + 2, 0x02, MSG_TYPE_TASK)

        v1_msg = v1_header + payload
        v2_msg = v2_header + payload

        assert v1_msg[4] == 0x01
        assert v2_msg[4] == 0x02
        assert v1_msg[4] != v2_msg[4]


class TestEndToEnd:
    """End-to-end tests simulating full executor -> worker -> executor flow."""

    def test_executor_to_worker_to_executor_inline(self, redis_client):
        """Simulate complete task flow with inline result."""
        from miniray.executor import Executor, MiniraySubTaskResult, _execute_batch
        from concurrent.futures import Future
        from unittest.mock import patch, Mock
        from functools import partial

        task_queue = "e2e-task-queue"
        result_queue = f"fq-{task_queue}"

        # === EXECUTOR SIDE: Submit task ===
        with patch('miniray.executor.redis.StrictRedis'):
            executor = object.__new__(Executor)
            executor.submit_queue_id = task_queue
            executor._futures = {}
            executor._submit_redis_master = Mock()

        # Create and pack task
        task_uuid = "e2e-uuid-001"
        fn = lambda x: x * 2
        pickled_fn = cloudpickle.dumps(partial(_execute_batch, fn))
        packed_task = executor._pack_task("", pickled_fn, [(21,)], {}, task_uuid)

        # Push to Redis (simulating executor._submit_task)
        redis_client.lpush(task_queue, packed_task)

        # === WORKER SIDE: Process task ===
        # Pop task from queue
        raw_task = redis_client.rpop(task_queue)
        assert raw_task is not None

        # Parse task (simulating worker.parse_task)
        _, version, msg_type = struct.unpack(">IBB", raw_task[:6])
        assert version == PROTOCOL_VERSION
        assert msg_type == MSG_TYPE_TASK

        task_payload = msgpack.unpackb(raw_task[6:], raw=False)
        assert task_payload["uuid"] == task_uuid

        # Execute task (simulating worker_task.py)
        func = cloudpickle.loads(task_payload["fn"])
        args, kwargs = cloudpickle.loads(task_payload["args"])
        result = func(*args, **kwargs)  # Returns ("__inline__", bytes) or (host, key)

        # Pack result (simulating worker.check_task_completion)
        result_type, result_data = result
        assert result_type == "__inline__"  # Small result should be inline

        result_msg = msgpack.packb({
            "job": task_queue,
            "worker": "test-worker",
            "task_uuid": task_uuid,
            "result": result_data,
        }, use_bin_type=True)
        header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INLINE)
        redis_client.lpush(result_queue, header + result_msg)

        # === EXECUTOR SIDE: Receive result ===
        future = Future()
        executor._futures[task_uuid] = [future]

        # Pop result (simulating executor._reader_loop with BLPOP)
        raw_result = redis_client.rpop(result_queue)
        assert raw_result is not None

        executor._unpack_result(raw_result)

        # Verify result
        assert future.done()
        assert future.result() == 42  # 21 * 2

    def test_executor_to_worker_to_executor_with_error(self, redis_client):
        """Simulate complete task flow where task raises an exception."""
        from miniray.executor import Executor, MiniraySubTaskResult, MinirayError, _execute_batch
        from concurrent.futures import Future
        from unittest.mock import patch, Mock
        from functools import partial

        task_queue = "e2e-error-queue"
        result_queue = f"fq-{task_queue}"

        # === EXECUTOR SIDE: Submit task ===
        with patch('miniray.executor.redis.StrictRedis'):
            executor = object.__new__(Executor)
            executor.submit_queue_id = task_queue
            executor._futures = {}
            executor._submit_redis_master = Mock()

        # Create task that will fail
        task_uuid = "e2e-error-uuid"
        def failing_fn(x):
            raise ValueError(f"intentional failure: {x}")

        pickled_fn = cloudpickle.dumps(partial(_execute_batch, failing_fn))
        packed_task = executor._pack_task("", pickled_fn, [(42,)], {}, task_uuid)
        redis_client.lpush(task_queue, packed_task)

        # === WORKER SIDE: Process task ===
        raw_task = redis_client.rpop(task_queue)
        task_payload = msgpack.unpackb(raw_task[6:], raw=False)

        func = cloudpickle.loads(task_payload["fn"])
        args, kwargs = cloudpickle.loads(task_payload["args"])
        result_type, result_data = func(*args, **kwargs)

        # Result should still be inline (contains error info)
        assert result_type == "__inline__"
        subtasks = cloudpickle.loads(result_data)
        assert len(subtasks) == 1
        assert subtasks[0].succeeded is False
        assert "intentional failure" in subtasks[0].exception_desc

        # Pack and send result
        result_msg = msgpack.packb({
            "job": task_queue,
            "worker": "test-worker",
            "task_uuid": task_uuid,
            "result": result_data,
        }, use_bin_type=True)
        header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INLINE)
        redis_client.lpush(result_queue, header + result_msg)

        # === EXECUTOR SIDE: Receive error ===
        future = Future()
        executor._futures[task_uuid] = [future]

        raw_result = redis_client.rpop(result_queue)
        executor._unpack_result(raw_result)

        assert future.done()
        with pytest.raises(MinirayError) as exc_info:
            future.result()
        assert "intentional failure" in str(exc_info.value)

    def test_executor_to_worker_to_executor_indirect(self, redis_client):
        """Simulate complete task flow with indirect (large) result."""
        from miniray.executor import Executor, MiniraySubTaskResult, INLINE_RESULT_THRESHOLD
        from concurrent.futures import Future
        from unittest.mock import patch, Mock

        task_queue = "e2e-indirect-queue"
        result_queue = f"fq-{task_queue}"
        payload_db = 10  # Worker stores payloads in db 10

        # Create a payload Redis client (simulating worker's local Redis)
        payload_client = redis.StrictRedis(
            host="localhost",
            port=redis_client.connection_pool.connection_kwargs['port'],
            db=payload_db
        )
        payload_client.flushdb()

        try:
            with patch('miniray.executor.redis.StrictRedis'):
                executor = object.__new__(Executor)
                executor.submit_queue_id = task_queue
                executor._futures = {}
                executor._submit_redis_master = Mock()

            # === Simulate worker creating indirect result ===
            task_uuid = "e2e-indirect-uuid"
            large_result = MiniraySubTaskResult(True, "", "", "x" * 1000)
            result_bytes = cloudpickle.dumps([large_result])

            # Store in "worker Redis" (db 10)
            payload_key = f"miniray-{task_uuid}"
            payload_client.lpush(payload_key, result_bytes)

            # Send indirect pointer through result queue
            result_msg = msgpack.packb({
                "job": task_queue,
                "worker": "test-worker",
                "task_uuid": task_uuid,
                "host": "localhost",
                "key": payload_key,
            }, use_bin_type=True)
            header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INDIRECT)
            redis_client.lpush(result_queue, header + result_msg)

            # === EXECUTOR SIDE: Receive indirect result ===
            future = Future()
            executor._futures[task_uuid] = [future]

            raw_result = redis_client.rpop(result_queue)

            # Patch Redis to return our payload client for the indirect fetch
            with patch('miniray.executor.redis.StrictRedis', return_value=payload_client):
                executor._unpack_result(raw_result)

            assert future.done()
            assert future.result() == "x" * 1000
        finally:
            payload_client.flushdb()
            payload_client.close()


class TestInlineThresholdBoundary:
    """Test behavior at the inline/indirect threshold boundary."""

    def test_result_just_under_threshold_is_inline(self):
        """Result just under 1MB should be inline."""
        from miniray.executor import _execute_batch, INLINE_RESULT_THRESHOLD

        # Create data that will be just under threshold after serialization
        # Account for MiniraySubTaskResult overhead (~100 bytes)
        target_size = INLINE_RESULT_THRESHOLD - 1000
        data = "x" * target_size

        def return_data():
            return data

        result_type, result_data = _execute_batch(return_data, ())

        assert result_type == "__inline__", f"Expected inline, got {result_type}"
        assert len(result_data) < INLINE_RESULT_THRESHOLD

    def test_result_over_threshold_is_indirect(self):
        """Result over 1MB should be indirect."""
        from miniray.executor import _execute_batch, INLINE_RESULT_THRESHOLD
        from unittest.mock import patch

        # Create data that will exceed threshold after serialization
        # Use 100KB margin to account for serialization overhead
        target_size = INLINE_RESULT_THRESHOLD + 100000
        data = "x" * target_size

        def return_data():
            return data

        with patch('miniray.executor._wrap_result_local_redis') as mock_wrap:
            mock_wrap.return_value = ("worker-host", "result-key")
            result_type, result_data = _execute_batch(return_data, ())

        assert mock_wrap.called, "Expected _wrap_result_local_redis to be called for large result"
        assert result_type == "worker-host"
        assert result_data == "result-key"

    def test_threshold_constant_is_1mb(self):
        """Verify threshold constant is exactly 1MB."""
        from miniray.executor import INLINE_RESULT_THRESHOLD
        assert INLINE_RESULT_THRESHOLD == 1024 * 1024


class TestFailureInjection:
    """Test error handling and recovery scenarios."""

    def test_corrupt_message_header_handled(self):
        """Corrupt message header should be logged, not crash."""
        from miniray.executor import Executor
        from unittest.mock import patch, Mock
        import sys
        from io import StringIO

        with patch('miniray.executor.redis.StrictRedis'):
            executor = object.__new__(Executor)
            executor._futures = {}
            executor._submit_redis_master = Mock()

        # Various corrupt messages
        corrupt_messages = [
            b"",                    # Empty
            b"x",                   # Too short
            b"xxxxx",              # Still too short (need 6 bytes)
            b"\x00\x00\x00\x10\x99\x01" + b"x" * 10,  # Wrong version
        ]

        for msg in corrupt_messages:
            # Should not raise, just log
            executor._unpack_result(msg)

    def test_missing_payload_key_in_indirect_result(self, redis_client):
        """Missing payload in worker Redis should set exception on future."""
        from miniray.executor import Executor, MinirayError
        from concurrent.futures import Future
        from unittest.mock import patch, Mock

        with patch('miniray.executor.redis.StrictRedis'):
            executor = object.__new__(Executor)
            executor._futures = {}
            executor._submit_redis_master = Mock()

        task_uuid = "missing-payload-uuid"
        future = Future()
        executor._futures[task_uuid] = [future]

        # Create indirect result pointing to non-existent key
        result_msg = msgpack.packb({
            "job": "test",
            "worker": "worker-1",
            "task_uuid": task_uuid,
            "host": "localhost",
            "key": "nonexistent-key",
        }, use_bin_type=True)
        header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INDIRECT)
        message = header + result_msg

        # Create a Redis client that returns None for lpop (missing key)
        mock_redis = Mock()
        mock_redis.lpop.return_value = None

        with patch('miniray.executor.redis.StrictRedis', return_value=mock_redis):
            executor._unpack_result(message)

        assert future.done()
        with pytest.raises(MinirayError) as exc_info:
            future.result()
        assert "Did not find payload" in str(exc_info.value)

    def test_redis_connection_error_in_reader_loop(self):
        """Redis connection error should be caught and retried."""
        from miniray.executor import Executor
        from unittest.mock import patch, Mock, call
        import sys
        from io import StringIO

        with patch('miniray.executor.redis.StrictRedis'):
            executor = object.__new__(Executor)
            executor._futures = {}
            executor._result_redis = Mock()
            executor.result_queue_id = "test-queue"
            executor.expiry_check_timer = time.time() + 100  # Skip expiry check
            executor._shutdown_reader_thread = False

        # Make blpop raise ConnectionError first, then return None and shutdown
        call_count = [0]
        def blpop_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise redis.ConnectionError("Connection lost")
            executor._shutdown_reader_thread = True
            return None

        executor._result_redis.blpop.side_effect = blpop_side_effect

        # Capture stderr
        old_stderr = sys.stderr
        sys.stderr = StringIO()

        try:
            # Run with patched sleep to avoid 10s wait
            with patch('miniray.executor.time.sleep') as mock_sleep:
                executor._reader_loop()

            # Should have logged error and called sleep(10)
            mock_sleep.assert_called_with(10)
            stderr_output = sys.stderr.getvalue()
            assert "Redis connection error" in stderr_output
        finally:
            sys.stderr = old_stderr

    def test_blpop_timeout_triggers_expiry_check(self):
        """BLPOP timeout should allow expiry check to run."""
        from miniray.executor import Executor
        from unittest.mock import patch, Mock

        with patch('miniray.executor.redis.StrictRedis'):
            executor = object.__new__(Executor)
            executor._futures = {"old-task": []}
            executor._result_redis = Mock()
            executor._submit_redis_master = Mock()
            executor._submit_redis_tasks = Mock()
            executor.result_queue_id = "test-queue"
            executor.submit_queue_id = "task-queue"
            executor.expiry_check_timer = 0  # Force expiry check
            executor.no_work_found_cnt = 0
            executor._shutdown_reader_thread = False

        # Mock Redis responses
        executor._result_redis.blpop.side_effect = [None, None]  # Two timeouts
        executor._submit_redis_master.get.return_value = None  # No start data
        executor._submit_redis_tasks.llen.return_value = 0  # Empty queue

        call_count = [0]
        original_blpop = executor._result_redis.blpop.side_effect
        def counting_blpop(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                executor._shutdown_reader_thread = True
            return None

        executor._result_redis.blpop.side_effect = counting_blpop

        executor._reader_loop()

        # Expiry check should have been called (it checks _submit_redis_master.get)
        assert executor._submit_redis_master.get.called or executor.no_work_found_cnt > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
