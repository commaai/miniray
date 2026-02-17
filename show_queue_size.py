#!/usr/bin/env python
import os
import redis
from miniray import REMOTE_QUEUE

REDIS_HOST = os.environ.get("REDIS_HOST", "redis.comma.internal")
REDIS_DB = int(os.environ.get("REDIS_DB", "1"))

client = redis.StrictRedis(host=REDIS_HOST, port=6379, db=REDIS_DB, decode_responses=True)
all_keys = list(client.scan_iter(match="*"))

queue_keys = [k for k in all_keys if k.endswith(f"-{REMOTE_QUEUE}")]
other_keys = [k for k in all_keys if not k.endswith(f"-{REMOTE_QUEUE}")]

def print_queue(title: str, items: list[str]) -> None:
  print(title)
  if not items:
    return
  pipe = client.pipeline()
  for key in items:
    pipe.type(key)
    pipe.llen(key)
  results = pipe.execute(raise_on_error=False)
  for i, key in enumerate(items):
    t, length = results[2 * i], results[2 * i + 1]
    if t == "list":
      print(f"{key} {length}")

print_queue(f"pending tasks for {REMOTE_QUEUE}:", queue_keys)
if other_keys:
  print_queue("pending tasks for other queues:", other_keys)
