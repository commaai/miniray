#!/usr/bin/env python
import os
import sys
import json
import redis
from typing import cast
from miniray import REMOTE_QUEUE
from miniray.executor import JobMetadata, get_metadata_key, migrate_job_metadata

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

REDIS_HOST = os.environ.get("REDIS_HOST", "redis.comma.internal")
REDIS_DB = int(os.environ.get("REDIS_DB", "1"))

client = redis.StrictRedis(host=REDIS_HOST, port=6379, db=REDIS_DB, decode_responses=True)
all_keys = list(client.scan_iter(match="*"))

queue_keys = [k for k in all_keys if k.endswith(f"-{REMOTE_QUEUE}")]
other_keys = [k for k in all_keys if not k.endswith(f"-{REMOTE_QUEUE}")]

def highlight(value: object, *styles: str) -> str:
  force_color = os.environ.get("FORCE_COLOR")
  if "NO_COLOR" in os.environ or force_color == "0" or not (sys.stdout.isatty() or force_color is not None):
    return str(value)
  return f"{''.join(styles)}{value}{RESET}"

def format_job(key: str, length: int, metadata: JobMetadata | None) -> str:
  info = ""
  if metadata:
    info = (
      f" {highlight('| Priority:', DIM)} {highlight(metadata.priority, YELLOW)}"
      f" {highlight('| Executor:', DIM)} {highlight(metadata.executor, MAGENTA)}"
    )
  return f"{highlight(key, CYAN)} {highlight(length, BOLD, GREEN)}{info}"

def print_queue(title: str, items: list[str]) -> None:
  print(highlight(title, BOLD, CYAN))
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
      job = format_job(key, length, metadata)
      if metadata and metadata.job_group:
        groups.setdefault(metadata.job_group, []).append(job)
      else:
        ungrouped.append(job)

  for group, jobs in sorted(groups.items()):
    print(f"{highlight('Group:', BOLD)} {highlight(group, BOLD, CYAN)}")
    for job in jobs:
      print(f"  {job}")
  for job in ungrouped:
    print(job)

print_queue(f"pending tasks for {REMOTE_QUEUE}:", queue_keys)
if other_keys:
  print_queue("pending tasks for other queues:", other_keys)
