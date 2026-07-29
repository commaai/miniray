#!/usr/bin/env python
import os
import json
import redis
from typing import cast
from miniray import REMOTE_QUEUE
from miniray.executor import get_metadata_key, migrate_job_metadata

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

  groups: dict[str, list[str]] = {}
  ungrouped: list[str] = []
  for i, key in enumerate(items):
    t, length = results[2 * i], results[2 * i + 1]
    if t == "list":
      raw = client.get(get_metadata_key(key))
      metadata = migrate_job_metadata(json.loads(cast(str, raw))) if raw else None
      info = f" | Priority: {metadata.priority} | Executor: {metadata.executor}" if metadata else ""
      job = f"{key} {length}{info}"
      if metadata and metadata.job_group:
        groups.setdefault(metadata.job_group, []).append(job)
      else:
        ungrouped.append(job)

  for group, jobs in sorted(groups.items()):
    print(f"Group: {group}")
    for job in jobs:
      print(f"  {job}")
  for job in ungrouped:
    print(job)

print_queue(f"pending tasks for {REMOTE_QUEUE}:", queue_keys)
if other_keys:
  print_queue("pending tasks for other queues:", other_keys)
