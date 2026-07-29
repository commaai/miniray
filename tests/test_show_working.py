import json
from unittest.mock import MagicMock

import show_working
from miniray.executor import JobMetadata, TaskRecord, TaskState
from miniray.lib.helpers import Limits


def test_show_working_migrates_legacy_metadata(monkeypatch, capsys):
  task = TaskRecord(
    "task-id", "job", "executor", "function", "pickled-fn", "pickled-args",
    TaskState.WORKING, "worker", 0.0, 1.0,
  )
  metadata = JobMetadata(True, 3, "/code", "host", Limits().asdict(), {}, "")

  client = MagicMock()
  client.keys.return_value = ["tasks:job"]
  client.hscan_iter.return_value = [("task-id", json.dumps(task))]
  client.httl.return_value = [60]
  client.get.return_value = json.dumps(metadata[:-1])
  monkeypatch.setattr(show_working.redis, "StrictRedis", lambda **kwargs: client)

  show_working.show_working()

  assert "Job: job | Active tasks: 1 | Priority: 3" in capsys.readouterr().out
