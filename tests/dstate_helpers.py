import os
import signal
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import cast

import pytest
from redis import StrictRedis


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


def get_active_workers(queue_name: str) -> set[str]:
  redis_host = os.environ.get('REDIS_HOST', 'redis.comma.internal')
  r = StrictRedis(host=redis_host, port=6379, db=1)
  prefix = f"active:{queue_name}:"
  keys = cast(list[bytes], r.keys(f"{prefix}*"))
  return {key.decode().removeprefix(prefix) for key in keys}


def wait_for_active_workers(queue_name: str, timeout: float = 30.0) -> set[str]:
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    workers = get_active_workers(queue_name)
    if workers:
      return workers
    time.sleep(0.5)
  pytest.fail(f"no active miniray workers on queue {queue_name}")


def wait_for_worker_to_disappear(queue_name: str, worker: str, timeout: float = 120.0):
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    if worker not in get_active_workers(queue_name):
      return
    time.sleep(0.5)
  pytest.fail(f"worker {worker} stayed active after the D-state task exceeded SIGKILL grace")
