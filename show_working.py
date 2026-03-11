#!/usr/bin/env python
import os
import sys
import json
import redis
from collections import defaultdict
from itertools import batched
from typing import cast
from miniray.executor import JobMetadata, TaskRecord, TaskState, get_metadata_key

REDIS_HOST = os.environ.get("REDIS_HOST", "redis.comma.internal")

def show_working():
  client = redis.StrictRedis(host=REDIS_HOST, port=6379, db=1, decode_responses=True)
  jobs: list[str] = cast(list[str], client.keys("tasks:*"))
  if not jobs:
    sys.exit(0)

  outputs = defaultdict(list)
  for job in jobs:
    for batch in batched(client.hscan_iter(job), 1000):
      records = {task_id: TaskRecord(*json.loads(task_value)) for task_id, task_value in batch}
      working = {task_id: record for task_id, record in records.items() if record.state == TaskState.WORKING}
      if working:
        ttls = cast(list[int], client.httl(job, *working.keys()))
        for record, ttl in zip(working.values(), ttls, strict=True):
          outputs[job].append(f"{record.worker:<24s} {record.executor:<24s} {ttl:8d}s remaining")

  job_priorities: dict[str, int] = {}
  for job in outputs:
    job_id = job.removeprefix('tasks:')
    raw = client.get(get_metadata_key(job_id))
    if raw:
      metadata = JobMetadata(*json.loads(raw))
      job_priorities[job] = metadata.priority

  for i, (job, lines) in enumerate(outputs.items()):
    job_id = job.removeprefix('tasks:')
    priority_str = f" | Priority: {job_priorities[job]}" if job in job_priorities else ""
    print()
    print(f"Job: {job_id} | Active tasks: {len(lines)}{priority_str}")
    print(f"{'Worker':<24s} {'Executor':<24s} {'TTL':>8s}")
    for line in lines:
      print(line)
    print()
    if i < len(outputs) - 1:
      print('-' * 60)

if __name__ == "__main__":
  show_working()
