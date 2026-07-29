from __future__ import annotations

import json
from collections import Counter
from unittest.mock import MagicMock
from lru import LRU

import worker
import miniray.executor as executor_module
from miniray.executor import JobMetadata, get_job_group_key, get_metadata_key
from miniray.lib.helpers import Limits


def test_worker_is_pinned_to_group_not_job(monkeypatch):
  # 50 jobs in groupA + 1 in groupB, equal priority, 4 workers.
  # Bundled into 2 groups -> 50/50 worker split, not groupA dominating with 50 slots.
  m: LRU[str, JobMetadata] = LRU(64)
  for j in [f"A_{i}" for i in range(50)]:
    m[j] = JobMetadata(True, 1, "/code", "host", Limits().asdict(), {}, "groupA")
  m["B_0"] = JobMetadata(True, 1, "/code", "host", Limits().asdict(), {}, "groupB")

  groups = worker.group_jobs(list(m.keys()), m)
  r = MagicMock()
  r.keys.return_value = [f"active:{worker.PIPELINE_QUEUE}:host{i}".encode() for i in range(4)]

  counts = Counter()
  for i in range(4):
    monkeypatch.setattr(worker, "ACTIVE_KEY", f"active:{worker.PIPELINE_QUEUE}:host{i}")
    group = worker.get_globally_scheduled_group(r, groups, m)
    assert group is not None
    counts[group] += 1
  assert counts == {"groupA": 2, "groupB": 2}, counts


def test_random_scheduler_excludes_groups_containing_gpu_jobs():
  m: LRU[str, JobMetadata] = LRU(64)
  m["mixed_cpu"] = JobMetadata(True, 1, "/code", "host", Limits().asdict(), {}, "mixed")
  m["mixed_gpu"] = JobMetadata(
    True, 1, "/code", "host", Limits(big_gpu_memory=1).asdict(), {}, "mixed")
  m["cpu_only"] = JobMetadata(True, 1, "/code", "host", Limits().asdict(), {}, "cpu_only")

  groups = worker.group_jobs(list(m.keys()), m)
  assert worker.get_randomly_scheduled_group(groups, m) == "cpu_only"
  assert worker.get_randomly_scheduled_group({"mixed": groups["mixed"]}, m) is None


def test_executor_writes_backward_compatible_job_metadata(monkeypatch, tmp_path):
  redis = MagicMock()
  redis.keys.return_value = [b"active-worker"]
  monkeypatch.setattr(executor_module, "StrictRedis", lambda **kwargs: redis)

  executor = executor_module.Executor(job_name="compat", job_group="group", codedir=str(tmp_path))
  set_values = {call.args[0]: call.args[1] for call in redis.set.call_args_list}
  raw_metadata = json.loads(set_values[get_metadata_key(executor.submit_queue_id)])

  assert len(raw_metadata) == 6
  assert JobMetadata(*raw_metadata).job_group == ""
  assert set_values[get_job_group_key(executor.submit_queue_id)] == "group"


def test_worker_loads_separate_job_group_and_legacy_metadata():
  job = "job-remote_v3"
  metadata = JobMetadata(True, 1, "/code", "host", Limits().asdict(), {}, "inline_group")

  job_metadatas: LRU[str, JobMetadata] = LRU(64)
  job_errors: LRU[str, tuple[str, str] | None] = LRU(64)
  redis = MagicMock()
  redis.get.side_effect = [json.dumps(metadata[:-1]).encode(), b"separate_group"]
  worker.update_job_metadatas(redis, [job], job_metadatas, job_errors)
  assert job_metadatas[job].job_group == "separate_group"

  job_metadatas.clear()
  redis.get.side_effect = [json.dumps(metadata[:-1]).encode(), None]
  worker.update_job_metadatas(redis, [job], job_metadatas, job_errors)
  assert job_metadatas[job].job_group == ""

  job_metadatas.clear()
  redis.get.side_effect = [json.dumps(metadata).encode(), None]
  worker.update_job_metadatas(redis, [job], job_metadatas, job_errors)
  assert job_metadatas[job].job_group == "inline_group"
