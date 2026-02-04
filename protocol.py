# Protocol V2 constants and helpers
# Shared between executor.py, worker.py, and tests

import struct
import msgpack

PROTOCOL_VERSION = 0x02
MSG_TYPE_TASK = 0x01
MSG_TYPE_RESULT_INLINE = 0x02
MSG_TYPE_RESULT_INDIRECT = 0x03
MSG_TYPE_RESULT_ERROR = 0x04
INLINE_RESULT_THRESHOLD = 1024 * 1024  # 1MB - results smaller than this go inline
BLPOP_TIMEOUT = 5  # seconds - timeout for blocking Redis read


def pack_task(submit_queue_id: str, function_ptr: str, pickled_fn: bytes,
              pickled_args: bytes, task_uuid: str) -> bytes:
    """Pack a task into V2 protocol format."""
    payload = msgpack.packb({
        "uuid": task_uuid,
        "job": submit_queue_id,
        "function_ptr": function_ptr,
        "fn": pickled_fn,
        "args": pickled_args,
    }, use_bin_type=True)
    header = struct.pack(">IBB", len(payload) + 2, PROTOCOL_VERSION, MSG_TYPE_TASK)
    return header + payload


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
