#!/usr/bin/env python
from __future__ import annotations

import os
import json
import sys
from collections import defaultdict
import redis
from typing import cast
from miniray import REMOTE_QUEUE
from miniray.executor import JobMetadata, get_metadata_key, migrate_job_metadata

REDIS_HOST = os.environ.get("REDIS_HOST", "redis.comma.internal")
REDIS_DB = int(os.environ.get("REDIS_DB", "1"))

def get_job_metadata(client: redis.StrictRedis, key: str) -> JobMetadata | None:
  raw = client.get(get_metadata_key(key))
  if raw:
    return migrate_job_metadata(json.loads(cast(str, raw)))
  return None

def format_job(key: str, length: int, metadata: JobMetadata | None) -> str:
  info = f" | Priority: {metadata.priority} | Executor: {metadata.executor}" if metadata else ""
  return f"{key} {length}{info}"

def format_group(group: str, color: bool) -> str:
  label = f"Group: {group}"
  return f"\033[1;36m{label}\033[0m" if color else label

def print_queue(client: redis.StrictRedis, title: str, items: list[str], color: bool) -> None:
  print(title)
  if not items:
    return
  pipe = client.pipeline()
  for key in items:
    pipe.type(key)
    pipe.llen(key)
  results = pipe.execute(raise_on_error=False)

  grouped_jobs: defaultdict[str, list[tuple[str, int, JobMetadata]]] = defaultdict(list)
  ungrouped_jobs: list[tuple[str, int, JobMetadata | None]] = []
  for i, key in enumerate(items):
    t, length = results[2 * i], results[2 * i + 1]
    if t == "list":
      metadata = get_job_metadata(client, key)
      job = (key, cast(int, length), metadata)
      if metadata is not None and metadata.job_group:
        grouped_jobs[metadata.job_group].append((key, cast(int, length), metadata))
      else:
        ungrouped_jobs.append(job)

  for group, jobs in sorted(grouped_jobs.items()):
    print(f"  {format_group(group, color)}")
    for key, length, metadata in sorted(jobs):
      print(f"    {format_job(key, length, metadata)}")
  for key, length, metadata in sorted(ungrouped_jobs):
    print(f"  {format_job(key, length, metadata)}")

def main() -> None:
  client = redis.StrictRedis(host=REDIS_HOST, port=6379, db=REDIS_DB, decode_responses=True)
  all_keys = list(client.scan_iter(match="*"))
  queue_keys = sorted(k for k in all_keys if k.endswith(f"-{REMOTE_QUEUE}"))
  other_keys = sorted(k for k in all_keys if not k.endswith(f"-{REMOTE_QUEUE}"))
  color = sys.stdout.isatty() and "NO_COLOR" not in os.environ

  print_queue(client, f"pending tasks for {REMOTE_QUEUE}:", queue_keys, color)
  if other_keys:
    print_queue(client, "pending tasks for other queues:", other_keys, color)

if __name__ == "__main__":
  main()
