import os
import time
import traceback
import types
import resource
import threading
import pynvml
from ctypes import _Pointer
from dataclasses import dataclass

from miniray.lib.triton_helpers import cleanup_triton
from miniray.lib.helpers import Limits, GB_TO_BYTES


class ResourceLimitError(Exception):
  pass


def check_gpu_status_worker(gpu_bus_ids, output):
  pynvml.nvmlInit()
  while True:
    try:
      for bus_id in gpu_bus_ids:
        gpu_dev = pynvml.nvmlDeviceGetHandleByPciBusId(bus_id)
        pynvml.nvmlDeviceGetTemperature(gpu_dev, pynvml.NVML_TEMPERATURE_GPU)  # sometimes we need to actually query the gpu to tell if it's fallen off
      output.valid = True
    except pynvml.NVMLError:
      output.valid = False
    except Exception:
      traceback.print_exc()
    finally:
      output.last_reading = time.time()
      time.sleep(10)


@dataclass
class GPUInfo:
    index: int
    memory: int
    bus_id: str
    small: bool
    handle: _Pointer[pynvml.struct_c_nvmlDevice_t]


@dataclass
class TaskAllocation:
    limits: Limits
    numa_node: int
    small_gpu_id: int | None
    big_gpu_id: int | None


class ResourceManager():
  def __init__(self, mem_limit_multiplier=0.8, triton_client=None):
    self._triton_client = triton_client
    self.gpu_status = types.SimpleNamespace(valid=True, last_reading=time.time())

    self.cpu_totals = self._get_cpu_info_by_node()
    self.mem_totals = self._get_mem_info_by_node(mem_limit_multiplier)
    self.gpus = self._get_gpu_info()
    self.big_gpus = [gpu for gpu in self.gpus if not gpu.small]
    self.small_gpus = [gpu for gpu in self.gpus if gpu.small]

    self._tasks: dict[str, TaskAllocation] = {}
    self.gpu_locked_job: str | None = None

    if self.gpus:
      thread = threading.Thread(target=check_gpu_status_worker, args=([gpu.bus_id for gpu in self.gpus], self.gpu_status))
      thread.daemon = True
      thread.start()

    # top level hard memory limit (covers child processes, too)
    resource.setrlimit(resource.RLIMIT_AS, (int(sum(self.mem_totals.values())), int(sum(self.mem_totals.values()))))

  def _get_cpu_usage_by_node(self) -> dict[int, float]:
    usage = dict.fromkeys(self.cpu_totals, 0.0)
    for alloc in self._tasks.values():
      usage[alloc.numa_node] += alloc.limits.cpu_threads
    return usage

  def _get_mem_usage_by_node(self) -> dict[int, float]:
    usage = dict.fromkeys(self.mem_totals, 0.0)
    for alloc in self._tasks.values():
      usage[alloc.numa_node] += alloc.limits.memory * GB_TO_BYTES
    return usage

  def _get_gpu_mem_usage(self) -> dict[int, float]:
    usage = {gpu.index: 0.0 for gpu in self.gpus}
    for alloc in self._tasks.values():
      if alloc.small_gpu_id is not None:
        usage[alloc.small_gpu_id] += alloc.limits.small_gpu_memory * GB_TO_BYTES
      if alloc.big_gpu_id is not None:
        usage[alloc.big_gpu_id] += alloc.limits.big_gpu_memory * GB_TO_BYTES
    return usage

  def _get_triton_users(self) -> int:
    return sum(1 for alloc in self._tasks.values() if alloc.limits.triton)

  def _has_active_gpu_job(self) -> bool:
    gpu_mem_usage = self._get_gpu_mem_usage()
    return any(usage > 0 for usage in gpu_mem_usage.values()) or self._get_triton_users() > 0

  def get_numa_node(self, task_uuid: str) -> int:
    return self._tasks[task_uuid].numa_node

  def get_gpu_ids(self, task_uuid: str) -> tuple[int | None, int | None]:
    alloc = self._tasks[task_uuid]
    return alloc.big_gpu_id, alloc.small_gpu_id

  def get_limits(self, task_uuid: str) -> Limits:
    return self._tasks[task_uuid].limits

  def consume(self, limits: Limits, job: str, task_uuid: str) -> None:
    if self.gpus:
      try:
        # don't start tasks if the gpus are not responding, or if the gpu status reading is stale
        if time.time() > self.gpu_status.last_reading + 20:
          raise Exception("waiting for gpu status reading...")
        elif not self.gpu_status.valid:
          raise Exception("unable to read gpu status")
      except Exception as e:
        raise ResourceLimitError(str(e)) from e

    mem_bytes = limits.memory * GB_TO_BYTES
    small_gpu_mem_bytes = limits.small_gpu_memory * GB_TO_BYTES
    big_gpu_mem_bytes = limits.big_gpu_memory * GB_TO_BYTES

    if limits.requires_gpu() and self._has_active_gpu_job() and self.gpu_locked_job != job:
      raise ResourceLimitError(f"GPU is locked to job {self.gpu_locked_job} cannot accept {job}")

    cpu_usages = self._get_cpu_usage_by_node()
    mem_usages = self._get_mem_usage_by_node()
    gpu_mem_usages = self._get_gpu_mem_usage()

    big_gpu, small_gpu = None, None
    if big_gpu_mem_bytes > 0 and self.big_gpus:
      big_gpu = min(self.big_gpus, key=lambda gpu: gpu_mem_usages[gpu.index])  # Pick the GPU with the lowest memory usage
    if small_gpu_mem_bytes > 0 and (self.small_gpus or self.big_gpus):
      small_gpu = min(self.small_gpus or self.big_gpus, key=lambda gpu: gpu_mem_usages[gpu.index])  # Fall back to big GPUs if no small ones are available

    if limits.triton and self._triton_client is None:
      raise ResourceLimitError("Triton client is not available for this ResourceManager")
    candidate_nodes = [node for node in self.cpu_totals if limits.cpu_threads <= self.cpu_totals[node] - cpu_usages[node]]
    if not candidate_nodes:
      raise ResourceLimitError(f"CPU request of {limits.cpu_threads} will exceed limit of {sum(self.cpu_totals.values())}")
    candidate_nodes = [node for node in candidate_nodes if (mem_bytes) <= self.mem_totals[node] - mem_usages[node]]
    if not candidate_nodes:
      raise ResourceLimitError(f"memory request of {mem_bytes} will exceed limit of {sum(self.mem_totals.values())}")
    numa_node = min(candidate_nodes, key=lambda node: cpu_usages[node] / self.cpu_totals[node])  # Pick the candidate node with the lowest CPU usage

    if small_gpu_mem_bytes and (not small_gpu or gpu_mem_usages[small_gpu.index] + small_gpu_mem_bytes > small_gpu.memory):
      raise ResourceLimitError("small gpu memory request of {} will exceed limit of {}".format(small_gpu_mem_bytes, small_gpu.memory if small_gpu else 0.0))
    if big_gpu_mem_bytes and (not big_gpu or gpu_mem_usages[big_gpu.index] + big_gpu_mem_bytes > big_gpu.memory):
      raise ResourceLimitError("big gpu memory request of {} will exceed limit of {}".format(big_gpu_mem_bytes, big_gpu.memory if big_gpu else 0.0))

    # Store allocation (no exceptions should be raised below this line)
    if self.gpu_locked_job != job and limits.requires_gpu():
      self.gpu_locked_job = job
      if self._triton_client is not None:
        cleanup_triton(self._triton_client, [gpu.bus_id for gpu in self.gpus])

    self._tasks[task_uuid] = TaskAllocation(
      limits=limits,
      numa_node=numa_node,
      small_gpu_id=small_gpu.index if small_gpu else None,
      big_gpu_id=big_gpu.index if big_gpu else None
    )

  def rekey(self, old_key: str, new_key: str) -> None:
    self._tasks[new_key] = self._tasks.pop(old_key)

  def release(self, task_uuid: str) -> None:
    if task_uuid in self._tasks:
      del self._tasks[task_uuid]

  def get_utilization(self):
    cpu_usages = self._get_cpu_usage_by_node()
    mem_usages = self._get_mem_usage_by_node()
    gpu_mem_usages = self._get_gpu_mem_usage()

    cpu_usage = sum(cpu_usages.values()) / sum(self.cpu_totals.values())
    mem_usage = sum(mem_usages.values()) / sum(self.mem_totals.values())
    small_gpu_mem_usage = sum(gpu_mem_usages[gpu.index] for gpu in self.small_gpus) / (sum(gpu.memory for gpu in self.small_gpus) + 1e-5)
    big_gpu_mem_usage = sum(gpu_mem_usages[gpu.index] for gpu in self.big_gpus) / (sum(gpu.memory for gpu in self.big_gpus) + 1e-5)
    return cpu_usage, mem_usage, small_gpu_mem_usage, big_gpu_mem_usage

  def _get_cpu_info_by_node(self):
    cpu_info = {}
    for d in os.listdir('/sys/devices/system/node/'):
      if d.startswith('node'):
        numa_node = int(d[4:])
        with open(f"/sys/devices/system/node/node{numa_node}/cpumap", "r") as f:
          cpu_bit_mask = f.read().strip().replace(",", "")
          bit_count = bin(int(cpu_bit_mask, 16)).count("1")
          cpu_info[numa_node] = bit_count
    return cpu_info

  def _get_mem_info_bytes(self, numa_node, k):
    """
    MemTotal:       65855368 kB
    MemAvailable:   63456920 kB
    """
    meminfo_fn = f"/sys/devices/system/node/node{numa_node}/meminfo"
    with open(meminfo_fn,'r') as f:
      for line in f:
        if f"{k}:" in line:
          # convert kb to bytes
          return int(line.strip().split()[-2]) * 1024
    raise LookupError(k)

  def _get_mem_info_by_node(self, mem_limit_multiplier):
    mem_info = {}
    for d in os.listdir('/sys/devices/system/node/'):
      if d.startswith('node'):
        numa_node = int(d[4:])
        mem_info[numa_node] = int(self._get_mem_info_bytes(numa_node, "MemTotal") * mem_limit_multiplier)
    return mem_info

  def _get_gpu_info(self):
    gpu_info = []
    try:
      pynvml.nvmlInit()
      for i in range(pynvml.nvmlDeviceGetCount()):
        gpu_dev = pynvml.nvmlDeviceGetHandleByIndex(i)
        pci_info = pynvml.nvmlDeviceGetPciInfo(gpu_dev)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_dev)
        is_small = "T600" in pynvml.nvmlDeviceGetName(gpu_dev)
        gpu_info.append(GPUInfo(index=i, memory=mem_info.total, bus_id=pci_info.busId, small=is_small, handle=gpu_dev))
    except pynvml.NVMLError_LibraryNotFound:
      print("WARNING: NVML shared library not found, running without GPUs")
      return []
    return gpu_info
