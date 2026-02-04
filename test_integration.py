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

# Protocol constants
PROTOCOL_VERSION = 0x02
MSG_TYPE_TASK = 0x01
MSG_TYPE_RESULT_INLINE = 0x02
MSG_TYPE_RESULT_INDIRECT = 0x03
MSG_TYPE_RESULT_ERROR = 0x04


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
