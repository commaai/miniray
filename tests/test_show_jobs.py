import json
from unittest.mock import MagicMock

import show_jobs
from miniray.executor import JobMetadata
from miniray.lib.helpers import Limits


def test_print_queue_groups_jobs_and_migrates_legacy_metadata(capsys):
  grouped_a = JobMetadata(True, 2, "/code", "host-a", Limits().asdict(), {}, "shared")
  grouped_b = JobMetadata(True, 1, "/code", "host-b", Limits().asdict(), {}, "shared")
  legacy = JobMetadata(True, 3, "/code", "host-c", Limits().asdict(), {}, "")

  client = MagicMock()
  client.pipeline.return_value.execute.return_value = ["list", 1, "list", 2, "list", 3]
  client.get.side_effect = [json.dumps(grouped_b), json.dumps(legacy[:-1]), json.dumps(grouped_a)]

  show_jobs.print_queue(client, "pending:", ["grouped-b", "legacy", "grouped-a"], color=False)

  assert capsys.readouterr().out == (
    "pending:\n"
    "  Group: shared\n"
    "    grouped-a 3 | Priority: 2 | Executor: host-a\n"
    "    grouped-b 1 | Priority: 1 | Executor: host-b\n"
    "  legacy 2 | Priority: 3 | Executor: host-c\n"
  )
