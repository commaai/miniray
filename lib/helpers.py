import re
import sys
import logging
from dataclasses import dataclass, asdict

# Memory conversion constants
GB_TO_BYTES = 1024 ** 3

# Worker configuration constants
TASK_TIMEOUT_GRACE_SECONDS = 10
MEMORY_LIMIT_HEADROOM = 1.2  # 20% extra to account for overhead
JOB_CACHE_SIZE = 1024
JOB_BLOCK_SECONDS = 60 * 5  # 5 minutes


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

def StreamLogger(name, level=None):
  logger = get_logger(name, level=level)
  if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler(sys.stdout))
  return logger


def desc(e):
  return f"{type(e).__name__}: {str(e)}"


@dataclass
class Limits:
  cpu_threads: int = 1
  memory: float = 1.0
  small_gpu_memory: float = 0.0
  big_gpu_memory: float = 0.0
  timeout_seconds: int = 60
  triton: bool = False

  def asdict(self):
    return asdict(self)

  def requires_gpu(self) -> bool:
    return self.small_gpu_memory > 0 or self.big_gpu_memory > 0 or self.triton


def extract_error(e):
  a = e.strip().split('\n')
  l = a[-1].split(':', 1)
  c = l[0].split('.')[-1]
  m = l[1].strip() if len(l) > 1 else ""
  try:
    if c == "AssertionError":
      m = a[-2].strip()
    elif c == "ColumnStoreException":
      pattern = r"\('(.*?)/[a-f0-9]+\|[0-9\-]+/[0-9]+/columnstore', 404\)"
      match = re.match(pattern, m)
      if match:
        m = match.groups(1)[0]
    elif c == "DependencyMissingError":
      m = m.split('/')[-3].strip()
    elif c == "Exception":
      patterns = ["Error, range out of bounds [0-9]{3}", "Error, requested range but got unexpected response [0-9]{3}", "Error [0-9]{3}"]
      for pattern in patterns:
        match = re.match(pattern, m)
        if match:
          m = str(match.group())
          break
  except IndexError:
    pass
  return f"{c}: {m}"
