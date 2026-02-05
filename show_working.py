#!/usr/bin/env python
import os
import json
import time
import redis

REDIS_HOST: str = os.environ.get("REDIS_HOST", "redis.comma.internal")

client = redis.StrictRedis(host=REDIS_HOST, port=6379, db=1, decode_responses=True)
keys: list[str] = sorted(client.scan_iter(match="*-start"))
if not keys:
  raise SystemExit(0)

pipe = client.pipeline()
for key in keys:
  pipe.get(key)
vals: list[str | None] = pipe.execute()

now = time.time()
for key, value in zip(keys, vals, strict=True):
  if value is None:
    continue
  job, worker, expiry_time = json.loads(value)
  remaining = float(expiry_time) - now
  task_uuid = key.removesuffix("-start")
  print(f"{worker:>10s} {job:>40s} {remaining:8.0f}s remaining")

