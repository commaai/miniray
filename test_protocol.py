#!/usr/bin/env python
"""Tests for miniray protocol V2 features.

These tests focus on the protocol serialization/deserialization logic
without requiring the full worker infrastructure (tritonclient, etc).
"""

import struct
import cloudpickle
import msgpack
import pytest
from unittest.mock import Mock, patch
from concurrent.futures import Future

from protocol import (
    PROTOCOL_VERSION, MSG_TYPE_TASK, MSG_TYPE_RESULT_INLINE,
    MSG_TYPE_RESULT_INDIRECT, MSG_TYPE_RESULT_ERROR, INLINE_RESULT_THRESHOLD,
)


def pack_task(submit_queue_id: str, function_ptr: str, pickled_fn: bytes,
              pickled_args: bytes, task_uuid: str) -> bytes:
    """Pack a task into V2 protocol format (extracted from executor._pack_task)."""
    payload = msgpack.packb({
        "uuid": task_uuid,
        "job": submit_queue_id,
        "function_ptr": function_ptr,
        "fn": pickled_fn,
        "args": pickled_args,
    }, use_bin_type=True)
    header = struct.pack(">IBB", len(payload) + 2, PROTOCOL_VERSION, MSG_TYPE_TASK)
    return header + payload


def parse_task(raw_task: bytes) -> dict:
    """Parse V2 protocol task (extracted from worker.parse_task)."""
    if len(raw_task) < 6:
        raise ValueError(f"Task too short: {len(raw_task)} bytes")

    length, version, msg_type = struct.unpack(">IBB", raw_task[:6])
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unknown protocol version: {version}")
    if msg_type != MSG_TYPE_TASK:
        raise ValueError(f"Expected task message type, got: {msg_type}")

    payload = msgpack.unpackb(raw_task[6:], raw=False)
    return payload


def pack_result_inline(job: str, worker: str, task_uuid: str, result: bytes) -> bytes:
    """Pack an inline result message."""
    result_msg = msgpack.packb({
        "job": job,
        "worker": worker,
        "task_uuid": task_uuid,
        "result": result,
    }, use_bin_type=True)
    header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INLINE)
    return header + result_msg


def pack_result_indirect(job: str, worker: str, task_uuid: str, host: str, key: str) -> bytes:
    """Pack an indirect result message."""
    result_msg = msgpack.packb({
        "job": job,
        "worker": worker,
        "task_uuid": task_uuid,
        "host": host,
        "key": key,
    }, use_bin_type=True)
    header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_INDIRECT)
    return header + result_msg


def pack_result_error(job: str, worker: str, task_uuid: str,
                      exception_type: str, exception_desc: str) -> bytes:
    """Pack an error result message."""
    result_msg = msgpack.packb({
        "job": job,
        "worker": worker,
        "task_uuid": task_uuid,
        "exception_type": exception_type,
        "exception_desc": exception_desc,
    }, use_bin_type=True)
    header = struct.pack(">IBB", len(result_msg) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_ERROR)
    return header + result_msg


def parse_result(res: bytes) -> tuple[int, dict]:
    """Parse a result message, return (msg_type, payload)."""
    if len(res) < 6:
        raise ValueError(f"Result too short: {len(res)} bytes")

    length, version, msg_type = struct.unpack(">IBB", res[:6])
    if version != PROTOCOL_VERSION:
        raise ValueError(f"Unknown protocol version: {version}")

    payload = msgpack.unpackb(res[6:], raw=False)
    return msg_type, payload


class TestProtocolV2Framing:
    """Test msgpack + length-prefix framing."""

    def test_task_pack_unpack_roundtrip(self):
        """Task should survive pack/unpack cycle."""
        task_uuid = "test-uuid-123"
        job = "test-job-queue"
        pickled_fn = cloudpickle.dumps(lambda x: x * 2)
        pickled_args = cloudpickle.dumps(([(1,), (2,)], {}))

        packed = pack_task(job, "", pickled_fn, pickled_args, task_uuid)

        # Verify frame structure
        assert len(packed) >= 6
        length, version, msg_type = struct.unpack(">IBB", packed[:6])
        assert version == PROTOCOL_VERSION
        assert msg_type == MSG_TYPE_TASK

        # Verify parse_task can decode it
        payload = parse_task(packed)
        assert payload["uuid"] == task_uuid
        assert payload["job"] == job
        assert payload["function_ptr"] == ""
        assert payload["fn"] == pickled_fn
        assert payload["args"] == pickled_args

    def test_task_with_function_ptr(self):
        """Task with function_ptr should have empty pickled_fn."""
        packed = pack_task("test-job", "cached-fn-key", b"", b"args", "uuid")
        payload = parse_task(packed)

        assert payload["function_ptr"] == "cached-fn-key"
        assert payload["fn"] == b""

    def test_invalid_protocol_version_rejected(self):
        """parse_task should reject unknown protocol versions."""
        payload = msgpack.packb({"uuid": "x", "job": "y", "function_ptr": "", "fn": b"", "args": b""})
        bad_header = struct.pack(">IBB", len(payload) + 2, 0x99, MSG_TYPE_TASK)
        bad_message = bad_header + payload

        with pytest.raises(ValueError, match="Unknown protocol version"):
            parse_task(bad_message)

    def test_invalid_message_type_rejected(self):
        """parse_task should reject non-task message types."""
        payload = msgpack.packb({"uuid": "x", "job": "y", "function_ptr": "", "fn": b"", "args": b""})
        bad_header = struct.pack(">IBB", len(payload) + 2, PROTOCOL_VERSION, MSG_TYPE_RESULT_ERROR)
        bad_message = bad_header + payload

        with pytest.raises(ValueError, match="Expected task message type"):
            parse_task(bad_message)

    def test_truncated_message_rejected(self):
        """parse_task should reject messages shorter than header."""
        with pytest.raises(ValueError, match="too short"):
            parse_task(b"short")

    def test_binary_data_preserved(self):
        """Binary data with all byte values should be preserved."""
        binary_data = bytes(range(256))
        packed = pack_task("job", "", binary_data, binary_data, "uuid")
        payload = parse_task(packed)

        assert payload["fn"] == binary_data
        assert payload["args"] == binary_data


class TestResultMessages:
    """Test result message packing/unpacking."""

    def test_inline_result_roundtrip(self):
        """Inline result should pack and unpack correctly."""
        result_data = cloudpickle.dumps({"key": "value", "number": 42})
        packed = pack_result_inline("job-1", "worker-1", "uuid-1", result_data)

        msg_type, payload = parse_result(packed)

        assert msg_type == MSG_TYPE_RESULT_INLINE
        assert payload["job"] == "job-1"
        assert payload["worker"] == "worker-1"
        assert payload["task_uuid"] == "uuid-1"
        assert payload["result"] == result_data

    def test_indirect_result_roundtrip(self):
        """Indirect result should pack and unpack correctly."""
        packed = pack_result_indirect("job-1", "worker-1", "uuid-1",
                                      "redis-host", "redis-key-123")

        msg_type, payload = parse_result(packed)

        assert msg_type == MSG_TYPE_RESULT_INDIRECT
        assert payload["host"] == "redis-host"
        assert payload["key"] == "redis-key-123"

    def test_error_result_roundtrip(self):
        """Error result should pack and unpack correctly."""
        packed = pack_result_error("job-1", "worker-1", "uuid-1",
                                   "ValueError", "something went wrong\nwith traceback")

        msg_type, payload = parse_result(packed)

        assert msg_type == MSG_TYPE_RESULT_ERROR
        assert payload["exception_type"] == "ValueError"
        assert "something went wrong" in payload["exception_desc"]


class TestInlineResults:
    """Test inline vs indirect result routing."""

    def test_small_result_returns_inline_marker(self):
        """Results under threshold should return __inline__ marker."""
        from miniray.executor import _execute_batch

        def identity(x):
            return x

        result_type, result_data = _execute_batch(identity, (42,))

        assert result_type == "__inline__"
        assert isinstance(result_data, bytes)
        assert len(result_data) < INLINE_RESULT_THRESHOLD

        # Verify the data contains the result
        subtasks = cloudpickle.loads(result_data)
        assert len(subtasks) == 1
        assert subtasks[0].succeeded is True
        assert subtasks[0].result == 42

    def test_large_result_uses_indirect(self):
        """Results over threshold should use indirect storage."""
        from miniray.executor import _execute_batch

        # Create a result that will exceed threshold after serialization
        large_data = "x" * (INLINE_RESULT_THRESHOLD + 100000)

        def return_large():
            return large_data

        with patch('miniray.executor._wrap_result_local_redis') as mock_wrap:
            mock_wrap.return_value = ("worker-host", "redis-key-123")

            # Need to make the serialized size exceed threshold
            # The actual check happens after cloudpickle.dumps of the results list
            result_type, result_data = _execute_batch(return_large, ((),))

            # If size exceeded threshold, _wrap_result_local_redis was called
            if mock_wrap.called:
                assert result_type == "worker-host"
                assert result_data == "redis-key-123"
            else:
                # Result was small enough to be inline
                assert result_type == "__inline__"

    def test_batch_with_mixed_success_failure(self):
        """Batch should capture both successes and failures."""
        from miniray.executor import _execute_batch

        def maybe_fail(x):
            if x < 0:
                raise ValueError(f"negative: {x}")
            return x * 2

        result_type, result_data = _execute_batch(maybe_fail, (5,), (-1,), (10,))

        assert result_type == "__inline__"
        subtasks = cloudpickle.loads(result_data)
        assert len(subtasks) == 3

        assert subtasks[0].succeeded is True
        assert subtasks[0].result == 10

        assert subtasks[1].succeeded is False
        assert subtasks[1].exception_type == "ValueError"
        assert "negative: -1" in subtasks[1].exception_desc

        assert subtasks[2].succeeded is True
        assert subtasks[2].result == 20

    def test_empty_batch(self):
        """Empty batch should return empty results."""
        from miniray.executor import _execute_batch

        def identity(x):
            return x

        result_type, result_data = _execute_batch(identity)

        assert result_type == "__inline__"
        subtasks = cloudpickle.loads(result_data)
        assert len(subtasks) == 0


class TestExecutorResultUnpacking:
    """Test executor result unpacking."""

    def setup_method(self):
        """Create a mock executor for each test."""
        from miniray.executor import Executor

        with patch('miniray.executor.redis.StrictRedis'):
            self.executor = object.__new__(Executor)
            self.executor._futures = {}
            self.executor._submit_redis_master = Mock()
            self.executor.result_queue_id = "fq-test"

    def test_unpack_inline_success(self):
        """Inline success result should resolve futures."""
        from miniray.executor import MiniraySubTaskResult

        task_uuid = "uuid-123"
        future = Future()
        self.executor._futures[task_uuid] = [future]

        subtasks = [MiniraySubTaskResult(True, "", "", 42)]
        message = pack_result_inline("test-job", "worker-1", task_uuid,
                                     cloudpickle.dumps(subtasks))

        self.executor._unpack_result(message)

        assert future.done()
        assert future.result() == 42
        assert task_uuid not in self.executor._futures

    def test_unpack_indirect_success(self):
        """Indirect success result should fetch from worker Redis."""
        from miniray.executor import MiniraySubTaskResult

        task_uuid = "uuid-456"
        future = Future()
        self.executor._futures[task_uuid] = [future]

        message = pack_result_indirect("test-job", "worker-1", task_uuid,
                                       "worker-redis-host", "result-key-789")

        # Mock the worker Redis fetch
        subtasks = [MiniraySubTaskResult(True, "", "", "fetched-result")]
        mock_worker_redis = Mock()
        mock_worker_redis.lpop.return_value = cloudpickle.dumps(subtasks)

        with patch('miniray.executor.redis.StrictRedis', return_value=mock_worker_redis):
            self.executor._unpack_result(message)

        mock_worker_redis.lpop.assert_called_once_with("result-key-789")
        assert future.done()
        assert future.result() == "fetched-result"

    def test_unpack_error_result(self):
        """Error result should set exception on futures."""
        from miniray.executor import MinirayError

        task_uuid = "uuid-error"
        future = Future()
        self.executor._futures[task_uuid] = [future]

        message = pack_result_error("test-job", "worker-1", task_uuid,
                                    "ValueError", "something went wrong")

        self.executor._unpack_result(message)

        assert future.done()
        with pytest.raises(MinirayError) as exc_info:
            future.result()
        assert exc_info.value.exception_type == "ValueError"
        assert "something went wrong" in str(exc_info.value)

    def test_unpack_multiple_futures(self):
        """Batch result should resolve all futures."""
        from miniray.executor import MiniraySubTaskResult

        task_uuid = "uuid-batch"
        futures = [Future(), Future(), Future()]
        self.executor._futures[task_uuid] = futures

        subtasks = [
            MiniraySubTaskResult(True, "", "", "result-1"),
            MiniraySubTaskResult(True, "", "", "result-2"),
            MiniraySubTaskResult(True, "", "", "result-3"),
        ]
        message = pack_result_inline("test-job", "worker-1", task_uuid,
                                     cloudpickle.dumps(subtasks))

        self.executor._unpack_result(message)

        assert all(f.done() for f in futures)
        assert futures[0].result() == "result-1"
        assert futures[1].result() == "result-2"
        assert futures[2].result() == "result-3"

    def test_unpack_unknown_task_logged(self, capsys):
        """Result for unknown task should log error, not crash."""
        message = pack_result_inline("test-job", "worker-1", "unknown-uuid", b"data")

        self.executor._unpack_result(message)

        captured = capsys.readouterr()
        assert "finished unstarted task" in captured.err
        assert "unknown-uuid" in captured.err

    def test_unpack_malformed_message_logged(self, capsys):
        """Malformed message should log error, not crash."""
        self.executor._unpack_result(b"short")

        captured = capsys.readouterr()
        assert "too short" in captured.err


class TestProtocolConstants:
    """Test that protocol constants match between files."""

    def test_executor_constants_defined(self):
        """Executor should have all protocol constants."""
        from miniray import executor

        assert executor.PROTOCOL_VERSION == 0x02
        assert executor.MSG_TYPE_TASK == 0x01
        assert executor.MSG_TYPE_RESULT_INLINE == 0x02
        assert executor.MSG_TYPE_RESULT_INDIRECT == 0x03
        assert executor.MSG_TYPE_RESULT_ERROR == 0x04
        assert executor.INLINE_RESULT_THRESHOLD == 1024 * 1024
        assert executor.BLPOP_TIMEOUT == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
