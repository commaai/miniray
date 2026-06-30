import os
import signal
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest
from redis import StrictRedis

GB_TO_BYTES = 1024 ** 3
WORKER_MEM_LIMIT_MULTIPLIER = 0.8


def _get_cpu_info_by_node() -> dict[int, int]:
  cpu_info = {}
  for entry in Path("/sys/devices/system/node").iterdir():
    if entry.name.startswith("node"):
      numa_node = int(entry.name[4:])
      cpu_bit_mask = (entry / "cpumap").read_text().strip().replace(",", "")
      cpu_info[numa_node] = bin(int(cpu_bit_mask, 16)).count("1")
  return cpu_info or {0: os.cpu_count() or 1}


def _get_mem_total_bytes(numa_node: int) -> int:
  with Path(f"/sys/devices/system/node/node{numa_node}/meminfo").open("r") as f:
    for line in f:
      if "MemTotal:" in line:
        return int(line.strip().split()[-2]) * 1024
  raise LookupError("MemTotal")


def get_worker_capacity(memory_gb: float, cpu_threads: int = 1) -> int:
  cpu_totals = _get_cpu_info_by_node()
  memory_bytes = int(memory_gb * GB_TO_BYTES)
  slots = 0
  for numa_node, cpu_total in cpu_totals.items():
    mem_total = int(_get_mem_total_bytes(numa_node) * WORKER_MEM_LIMIT_MULTIPLIER)
    slots += min(cpu_total // cpu_threads, mem_total // memory_bytes)
  return max(1, slots)


def block_in_frozen_filesystem(hold_seconds: int):
  token = uuid.uuid4().hex
  tmp_root = Path(tempfile.mkdtemp(prefix=f"miniray-dstate-{token}-", dir="/tmp"))
  img = tmp_root / "fs.img"
  mnt = tmp_root / "mnt"
  ready = tmp_root / "watchdog.ready"
  watchdog_cgroup = Path("/sys/fs/cgroup") / f"miniray-dstate-watchdog-{token}"
  watchdog_pid = None
  mounted = False

  def run_command(args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  def cleanup_mount():
    subprocess.run(["fsfreeze", "-u", str(mnt)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
      if subprocess.run(["umount", str(mnt)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        break
      time.sleep(0.1)
    shutil.rmtree(tmp_root, ignore_errors=True)
    try:
      watchdog_cgroup.rmdir()
    except OSError:
      pass

  def start_watchdog():
    pid = os.fork()
    if pid != 0:
      return pid

    try:
      os.setsid()
      signal.signal(signal.SIGTERM, signal.SIG_IGN)
      signal.signal(signal.SIGINT, signal.SIG_IGN)

      watchdog_cgroup.mkdir(exist_ok=True)
      cgroup_procs = watchdog_cgroup / "cgroup.procs"
      if cgroup_procs.exists():
        cgroup_procs.write_text(str(os.getpid()))
      ready.write_text(Path(f"/proc/{os.getpid()}/cgroup").read_text())
      time.sleep(hold_seconds)
    finally:
      cleanup_mount()
    os._exit(0)

  try:
    mnt.mkdir()
    run_command(["fallocate", "-l", "64M", str(img)])
    run_command(["mkfs.ext4", "-q", "-F", str(img)])
    run_command(["mount", "-o", "loop", str(img), str(mnt)])
    mounted = True

    watchdog_pid = start_watchdog()
    for _ in range(1000):
      if ready.exists():
        break
      time.sleep(0.01)
    else:
      raise RuntimeError("D-state watchdog did not report readiness")

    if watchdog_cgroup.name not in ready.read_text():
      raise RuntimeError("D-state watchdog did not move out of the task cgroup")

    run_command(["fsfreeze", "-f", str(mnt)])
    (mnt / "blocked-dir").mkdir()
    raise RuntimeError("mkdir on a frozen filesystem unexpectedly completed")
  finally:
    if watchdog_pid is not None:
      try:
        os.kill(watchdog_pid, signal.SIGKILL)
        os.waitpid(watchdog_pid, 0)
      except ChildProcessError:
        pass
      except ProcessLookupError:
        pass
    if mounted:
      cleanup_mount()


def wait_for_worker_to_disappear(queue_name: str, worker: str, timeout: float = 120.0):
  if not worker:
    pytest.fail("lost task did not report a worker")

  redis_host = os.environ.get('REDIS_HOST', 'redis.comma.internal')
  r = StrictRedis(host=redis_host, port=6379, db=1)
  active_key = f"active:{queue_name}:{worker}"
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if not r.exists(active_key):
      return
    time.sleep(0.5)
  pytest.fail(f"worker {worker} stayed active after the D-state task exceeded SIGKILL grace")
