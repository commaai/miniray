#!/usr/bin/env python
import os
import sys
import json
import redis
import itertools
from typing import cast
from miniray.executor import TaskRecord, TaskState

REDIS_HOST = os.environ.get("REDIS_HOST", "redis.comma.internal")

def batched(iterable, n):
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch

def show_working():
  client = redis.StrictRedis(host=REDIS_HOST, port=6379, db=1, decode_responses=True)
  jobs: list[str] = cast(list[str], client.keys("tasks:*"))
  if not jobs:
    sys.exit(0)

  for i, job in enumerate(jobs):
    lines = []
    for batch in batched(client.hscan_iter(job), 1000):
      records = {task_id: TaskRecord(*json.loads(task_value)) for task_id, task_value in batch}
      working = {task_id: record for task_id, record in records.items() if record.state == TaskState.WORKING}
      if working:
        ttls = cast(list[int], client.httl(job, *working.keys()))
        for record, ttl in zip(working.values(), ttls, strict=True):
          lines.append(f"{record.worker:<24s} {record.executor:<24s} {ttl:8d}s remaining")

    print()
    print(f"Job: {job.removeprefix('tasks:')} | Active tasks: {len(lines)}")
    if lines:
      print(f"{'Worker':<24s} {'Executor':<24s} {'TTL':>8s}")
      for line in lines:
        print(line)
    print()

    if i < len(jobs) - 1:
      print('-' * 60)

if __name__ == "__main__":
  show_working()
