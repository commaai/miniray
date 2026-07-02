#!/usr/bin/env python
import os
import sys
import io
import json
import traceback
import cloudpickle
import wrapt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling imports must work from any job venv
from landlock import restrict_file_writes
from task_sandbox import sandbox_uses_landlock, sandbox_write_paths

# Tasks initially start as root so they can be moved into the appropriate cgroup
os.setuid(int(os.getenv("TASK_UID", 1000)))


def apply_sandbox():
  # Deny filesystem writes outside the task's scratch directories; reads and execution stay
  # unrestricted. The sandbox is inherited by everything the task spawns and cannot be lifted.
  # Jobs can whitelist extra paths via MINIRAY_SANDBOX_WRITE_PATHS (colon-separated).
  read_write_paths = sandbox_write_paths(create_pycache=True)
  if not restrict_file_writes(read_write_paths, write_file_paths=["/dev"]):  # /dev: GPU nodes, /dev/null, ...
    print("[worker_task] Landlock unsupported by kernel, task runs without write sandbox", file=sys.stderr)


if sandbox_uses_landlock():
  apply_sandbox()

@wrapt.when_imported("cv2")
def cv2_import_hook(cv2):
  cv2.setNumThreads(0)

@wrapt.when_imported("torch")
def torch_import_hook(torch):
  torch.set_num_threads(1) # intraop parallelism
  torch.set_num_interop_threads(1)


def worker_process():
  os.nice(18)

  result_file = Path(os.environ['RESULT_FILE'])

  try:
    pickle_buffer = io.BytesIO(sys.stdin.buffer.read())
    func = cloudpickle.load(pickle_buffer)
    args, kwargs = cloudpickle.load(pickle_buffer)
    result = cloudpickle.dumps(func(*args, **kwargs))

    binary_out = b'\x00' + result
  except BaseException as e:
    error_info = (type(e).__name__, traceback.format_exc())
    binary_out = b'\x01' + json.dumps(error_info).encode()
  with result_file.open('wb') as f:
    f.write(binary_out)


if __name__ == '__main__':
  worker_process()
