#!/usr/bin/env python
import os
import sys
import io
import json
import traceback
import cloudpickle
import wrapt

# Tasks initially start as root so they can be moved into the appropriate cgroup
os.setuid(int(os.getenv("TASK_UID", 1000)))

@wrapt.when_imported("cv2")
def cv2_import_hook(cv2):
  cv2.setNumThreads(0)

@wrapt.when_imported("torch")
def torch_import_hook(torch):
  torch.set_num_threads(1) # intraop parallelism
  torch.set_num_interop_threads(1)


def worker_process():
  os.nice(18)

  result_file = os.environ['RESULT_FILE']

  try:
    pickle_buffer = io.BytesIO(sys.stdin.buffer.read())
    func = cloudpickle.load(pickle_buffer)
    args, kwargs = cloudpickle.load(pickle_buffer)
    result = cloudpickle.dumps(func(*args, **kwargs))

    binary_out = b'\x00' + result
  except BaseException as e:
    error_info = (type(e).__name__, traceback.format_exc())
    binary_out = b'\x01' + json.dumps(error_info).encode()
  with open(result_file, 'wb') as f:
    f.write(binary_out)


if __name__ == '__main__':
  worker_process()
