import os
import sys
import resource
from time import process_time

DOCKER_CONTAINER = os.path.exists("/.dockerenv")
TASK_CGROUP = os.getenv("TASK_CGROUP", "")

def get_cgroup_cpu_usage(cgroup):
  with open(f'/sys/fs/cgroup/{cgroup}/cpu.stat') as f:
    usage_usec = next(l for l in f if l.startswith("usage_usec")).split()[1]
    cpu_seconds = int(usage_usec) / 1e6
  return cpu_seconds

def get_cgroup_mem_usage(cgroup):
  with open(f'/sys/fs/cgroup/{cgroup}/memory.peak') as f:
    max_bytes = int(f.read())
  return max_bytes

def get_cpu_usage():
  if DOCKER_CONTAINER or TASK_CGROUP:
    return get_cgroup_cpu_usage(TASK_CGROUP)
  return process_time()

def get_mem_usage():
  if DOCKER_CONTAINER or TASK_CGROUP:
    return get_cgroup_mem_usage(TASK_CGROUP)

  maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if sys.platform != 'linux':
    raise Exception("rusage units unknown")
  return maxrss * 1024 # linux is in KB

def get_gpu_stats(pid, devices):
  import pynvml
  try:
    for device in devices:
      try:
        return pynvml.nvmlDeviceGetAccountingStats(device, pid)
      except pynvml.NVMLError_NotFound:
        continue
  except pynvml.NVMLError:
    pass
  return None

def get_gpu_mem_usage(stats):
  return stats.maxMemoryUsage if stats else 0

def get_gpu_utilization(stats):
  return stats.gpuUtilization / 100 if stats else 0
