#!/usr/bin/env python
import os
import redis
from miniray import REMOTE_QUEUE

REDIS_HOST = os.environ.get("REDIS_HOST", "redis.comma.internal")
REDIS_DB = int(os.environ.get("REDIS_DB", "1"))

client = redis.StrictRedis(host=REDIS_HOST, port=6379, db=REDIS_DB, decode_responses=True)
keys = [k for k in client.scan_iter(match="*") if ":" not in k]

queue_keys: dict[str, list[str]] = {REMOTE_QUEUE: []}
other_keys: list[str] = []

for key in keys:
  if key.endswith(f"-{REMOTE_QUEUE}"):
    queue_keys[REMOTE_QUEUE].append(key)
  else:
    other_keys.append(key)

def print_queue(title: str, items: list[str]) -> None:
  print(title)
  if not items:
    return
  pipe = client.pipeline()
  for key in items:
    pipe.llen(key)
  lengths = pipe.execute()
  for key, length in sorted(zip(items, lengths, strict=True)):
    print(f"{key} {length}")

print_queue(f"pending tasks for {REMOTE_QUEUE} :", queue_keys[REMOTE_QUEUE])

if other_keys:
  print_queue("pending tasks for other queues :", other_keys)
