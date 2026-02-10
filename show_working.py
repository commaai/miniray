#!/usr/bin/env python
import os
import json
import redis
from miniray.executor import TaskRecord

REDIS_HOST: str = os.environ.get("REDIS_HOST", "redis.comma.internal")

client = redis.StrictRedis(host=REDIS_HOST, port=6379, db=1, decode_responses=True)
keys: list[str] = sorted(client.scan_iter(match="task:*"))
if not keys:
  raise SystemExit(0)

pipe = client.pipeline()
for key in keys:
  pipe.get(key)
  pipe.ttl(key)
results: list = pipe.execute()

for i in range(0, len(results), 2):
  value = results[i]
  ttl = results[i + 1]
  if value is None:
    continue
  record = TaskRecord(*json.loads(value))
  if record.state != 'working':
    continue
  print(f"{record.worker:>10s} {record.job:>40s} {ttl:8d}s remaining")
