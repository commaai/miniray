import json
from typing import cast

from redis import StrictRedis

from miniray.executor import JobMetadata, get_metadata_key
from miniray.lib.helpers import JOB_CACHE_SIZE, Limits
from miniray.worker import make_job_metadata_stores, update_job_metadatas


class FakeRedis:
  def __init__(self, values: dict[str, bytes]):
    self.values = values

  def get(self, key: str) -> bytes | None:
    return self.values.get(key)


def make_metadata() -> JobMetadata:
  return JobMetadata(True, 1, "/code.nfs/xx", "executor", Limits().asdict(), {})


def test_job_metadata_store_keeps_all_active_jobs():
  jobs = [f"job_{i}-remote_v3" for i in range(JOB_CACHE_SIZE + 5)]
  raw_metadatas = {get_metadata_key(job): json.dumps(make_metadata()).encode() for job in jobs}
  stale_job = "stale_job-remote_v3"
  job_metadatas, job_errors = make_job_metadata_stores()
  job_metadatas[stale_job] = make_metadata()
  job_errors[stale_job] = None

  update_job_metadatas(cast(StrictRedis, FakeRedis(raw_metadatas)), jobs, job_metadatas, job_errors)

  assert set(job_metadatas) == set(jobs)
  assert set(job_errors) == set(jobs)
  assert all(job_errors[job] is None for job in jobs)
  assert [job for job in jobs if not job_metadatas[job].limits.get("node_whitelist")] == jobs
