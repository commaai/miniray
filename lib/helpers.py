import re
import sys
import logging
from typing import Optional
from dataclasses import dataclass, asdict

# Memory conversion constants
GB_TO_BYTES = 1024 ** 3

# Worker configuration constants
TASK_TIMEOUT_GRACE_SECONDS = 10
MAX_WORKER_LOOP_SECONDS = 10
JOB_CACHE_SIZE = 1024
JOB_BLOCK_SECONDS = 60 * 5  # 5 minutes

@dataclass
class Limits:
  cpu_threads: int = 1
  memory: float = 1.0
  small_gpu_memory: float = 0.0
  big_gpu_memory: float = 0.0
  timeout_seconds: int = 60
  triton: bool = False
  node_whitelist: Optional[list[str]] = None

  def asdict(self):
    return asdict(self)

  def requires_gpu(self) -> bool:
    return self.small_gpu_memory > 0 or self.big_gpu_memory > 0 or self.triton


def set_random_seeds(seed: int):
  import os
  import random
  import numpy as np
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)

def get_logger(name, level=None):
  logger = logging.getLogger(name)
  logger.setLevel(level or logging.DEBUG)
  logger.propagate = False
  return logger

def get_stream_logger(name, level=None):
  logger = get_logger(name, level=level)
  if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler(sys.stdout))
  return logger


def desc(e):
  return f"{type(e).__name__}: {str(e)}"

def extract_error(e):
  lines = e.strip().split('\n')
  last_line = lines[-1].split(':', 1)
  cls = last_line[0].split('.')[-1]
  msg = last_line[1].strip() if len(last_line) > 1 else ""
  try:
    if cls == "AssertionError":
      msg = lines[-2].strip()
    elif cls == "ColumnStoreException":
      pattern = r"\('(.*?)/[a-f0-9]+\|[0-9\-]+/[0-9]+/columnstore', 404\)"
      match = re.match(pattern, msg)
      if match:
        msg = match.groups(1)[0]
    elif cls == "DependencyMissingError":
      msg = msg.split('/')[-3].strip()
    elif cls == "Exception":
      patterns = ["Error, range out of bounds [0-9]{3}", "Error, requested range but got unexpected response [0-9]{3}", "Error [0-9]{3}"]
      for pattern in patterns:
        match = re.match(pattern, msg)
        if match:
          msg = str(match.group())
          break
  except IndexError:
    pass
  return f"{cls}: {msg}"
