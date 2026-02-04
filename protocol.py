# Protocol V2 constants
# Shared between executor.py, worker.py, and tests

PROTOCOL_VERSION = 0x02
MSG_TYPE_TASK = 0x01
MSG_TYPE_RESULT_INLINE = 0x02
MSG_TYPE_RESULT_INDIRECT = 0x03
MSG_TYPE_RESULT_ERROR = 0x04
INLINE_RESULT_THRESHOLD = 1024 * 1024  # 1MB - results smaller than this go inline
BLPOP_TIMEOUT = 5  # seconds - timeout for blocking Redis read
