from __future__ import annotations

import json
from collections import Counter
from unittest.mock import MagicMock
from lru import LRU

import worker
import miniray.executor as executor_module
from miniray.executor import JobMetadata, get_metadata_key
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


def test_unset_job_group_keeps_jobs_separate():
  m: LRU[str, JobMetadata] = LRU(64)
  m["same_name_a"] = JobMetadata(True, 1, "/code", "host", Limits().asdict(), {})
  m["same_name_b"] = JobMetadata(True, 1, "/code", "host", Limits().asdict(), {})

  assert worker.group_jobs(list(m.keys()), m) == {
    "same_name_a": ["same_name_a"],
    "same_name_b": ["same_name_b"],
  }


def test_job_config_preserves_positional_queue_name():
  config = executor_module.JobConfig(5, "name", "custom_queue")
  assert config.priority == 5
  assert config.job_name == "name"
  assert config.queue_name == "custom_queue"
  assert config.job_group == ""


def test_random_scheduler_excludes_groups_containing_gpu_jobs():
  m: LRU[str, JobMetadata] = LRU(64)
  m["mixed_cpu"] = JobMetadata(True, 1, "/code", "host", Limits().asdict(), {}, "mixed")
  m["mixed_gpu"] = JobMetadata(
    True, 1, "/code", "host", Limits(big_gpu_memory=1).asdict(), {}, "mixed")
  m["cpu_only"] = JobMetadata(True, 1, "/code", "host", Limits().asdict(), {}, "cpu_only")

  groups = worker.group_jobs(list(m.keys()), m)
  assert worker.get_randomly_scheduled_group(groups, m) == "cpu_only"
  assert worker.get_randomly_scheduled_group({"mixed": groups["mixed"]}, m) is None


def test_executor_writes_job_group_in_metadata(monkeypatch, tmp_path):
  redis = MagicMock()
  redis.keys.return_value = [b"active-worker"]
  monkeypatch.setattr(executor_module, "StrictRedis", lambda **kwargs: redis)

  executor = executor_module.Executor(job_name="compat", job_group="group", codedir=str(tmp_path))
  ungrouped_executor = executor_module.Executor(job_name="compat", codedir=str(tmp_path))
  set_values = {call.args[0]: call.args[1] for call in redis.set.call_args_list}
  grouped_metadata = JobMetadata(*json.loads(set_values[get_metadata_key(executor.submit_queue_id)]))
  ungrouped_metadata = JobMetadata(*json.loads(set_values[get_metadata_key(ungrouped_executor.submit_queue_id)]))

  assert grouped_metadata.job_group == "group"
  assert ungrouped_metadata.job_group == ""


def test_worker_loads_new_and_legacy_metadata():
  job = "job-remote_v3"
  metadata = JobMetadata(True, 1, "/code", "host", Limits().asdict(), {}, "inline_group")

  job_metadatas: LRU[str, JobMetadata] = LRU(64)
  job_errors: LRU[str, tuple[str, str] | None] = LRU(64)
  redis = MagicMock()
  redis.get.return_value = json.dumps(metadata[:-1]).encode()
  worker.update_job_metadatas(redis, [job], job_metadatas, job_errors)
  assert job_metadatas[job].job_group == ""

  job_metadatas.clear()
  redis.get.return_value = json.dumps(metadata).encode()
  worker.update_job_metadatas(redis, [job], job_metadatas, job_errors)
  assert job_metadatas[job].job_group == "inline_group"
