from __future__ import annotations

from unittest.mock import MagicMock
from collections import Counter
from lru import LRU

import worker
from miniray.executor import JobMetadata
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
