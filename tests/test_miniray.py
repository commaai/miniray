import os
import time
from pathlib import Path
import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

import miniray

MINIRAY_PRIORITY = 1000
MINIRAY_MEMORY_GB = 0.4
QUEUE_NAME = os.environ.get('MINIRAY_QUEUE', miniray.REMOTE_QUEUE)


class MinirayTestClass:
  def __init__(self, value):
    self.value = value
  def get_miniray_output(self, x):
    return self.value + x

def get_miniray_error():
  raise RuntimeError("Ruh roh!")

def is_even(n):
  return n % 2 == 0

def make_random_payload(size: int) -> bytes:
  return os.urandom(size)

def slow_sleep(seconds: float) -> str:
  time.sleep(seconds)
  return "done"

def spawn_zombie():
  pid = os.fork()
  if pid == 0:
    time.sleep(300)
    os._exit(0)
  return "done"

def remote_worker_supports_d_state_test():
  import os
  import shutil
  from pathlib import Path

  required_commands = ("bash", "fallocate", "fsfreeze", "mkfs.ext4", "mount", "umount")
  return (
    os.geteuid() == 0
    and all(shutil.which(cmd) is not None for cmd in required_commands)
    and Path("/sys/fs/cgroup").is_dir()
  )


def block_in_frozen_filesystem(hold_seconds: int):
  import shutil
  import subprocess
  import tempfile
  import time
  import uuid
  from pathlib import Path

  if not remote_worker_supports_d_state_test():
    raise RuntimeError("worker does not have root, cgroup, loop mount, and fsfreeze support")

  token = uuid.uuid4().hex
  tmp_root = Path(tempfile.mkdtemp(prefix=f"miniray-dstate-{token}-", dir="/tmp"))
  img = tmp_root / "fs.img"
  mnt = tmp_root / "mnt"
  ready = tmp_root / "watchdog.ready"
  watchdog = tmp_root / "watchdog.sh"
  watchdog_cgroup = Path("/sys/fs/cgroup") / f"miniray-dstate-watchdog-{token}"
  watchdog_proc = None
  mounted = False

  def run_command(args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  watchdog.write_text(f"""#!/usr/bin/env bash
set -u
mnt=\"{mnt}\"
img=\"{img}\"
tmp_root=\"{tmp_root}\"
ready=\"{ready}\"
watchdog_cgroup=\"{watchdog_cgroup}\"
hold_seconds=\"{hold_seconds}\"

cleanup() {{
  fsfreeze -u \"$mnt\" 2>/dev/null || true
  for _ in {{1..100}}; do
    umount \"$mnt\" 2>/dev/null && break
    sleep 0.1
  done
  rmdir \"$mnt\" 2>/dev/null || true
  rm -f \"$img\" \"$ready\" \"$0\" 2>/dev/null || true
  rmdir \"$tmp_root\" 2>/dev/null || true
  rmdir \"$watchdog_cgroup\" 2>/dev/null || true
}}
trap cleanup EXIT
trap "exit 0" TERM INT

mkdir \"$watchdog_cgroup\" 2>/dev/null || true
if [[ -w \"$watchdog_cgroup/cgroup.procs\" ]]; then
  echo \"$$\" > \"$watchdog_cgroup/cgroup.procs\" 2>/dev/null || true
fi
cat \"/proc/$$/cgroup\" > \"$ready\" 2>/dev/null || touch \"$ready\"
sleep \"$hold_seconds\"
""")
  watchdog.chmod(0o755)

  try:
    mnt.mkdir()
    run_command(["fallocate", "-l", "64M", str(img)])
    run_command(["mkfs.ext4", "-q", "-F", str(img)])
    run_command(["mount", "-o", "loop", str(img), str(mnt)])
    mounted = True

    watchdog_proc = subprocess.Popen(
      ["bash", str(watchdog)],
      stdin=subprocess.DEVNULL,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      start_new_session=True,
    )
    for _ in range(1000):
      if ready.exists():
        break
      if watchdog_proc.poll() is not None:
        raise RuntimeError(f"D-state watchdog exited early with code {watchdog_proc.returncode}")
      time.sleep(0.01)
    else:
      raise RuntimeError("D-state watchdog did not report readiness")

    if watchdog_cgroup.name not in ready.read_text():
      raise RuntimeError("D-state watchdog did not move out of the task cgroup")

    run_command(["fsfreeze", "-f", str(mnt)])
    (mnt / "blocked-dir").mkdir()
    raise RuntimeError("mkdir on a frozen filesystem unexpectedly completed")
  finally:
    if watchdog_proc is not None and watchdog_proc.poll() is None:
      watchdog_proc.terminate()
      try:
        watchdog_proc.wait(timeout=1)
      except subprocess.TimeoutExpired:
        watchdog_proc.kill()
    if mounted:
      subprocess.run(["fsfreeze", "-u", str(mnt)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      subprocess.run(["umount", str(mnt)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp_root, ignore_errors=True)
    try:
      watchdog_cgroup.rmdir()
    except OSError:
      pass


def get_executor(job_name: str) -> miniray.Executor:
  return miniray.Executor(job_name=job_name,
                          priority=MINIRAY_PRIORITY,
                          queue_name=QUEUE_NAME,
                          limits={'memory': MINIRAY_MEMORY_GB})

# Tests

def test_map_matches_local_and_threadpool():
  args = np.arange(100)
  results_loop = [is_even(n) for n in args]

  with ThreadPoolExecutor(max_workers=8) as executor:
    results_threadpool = list(executor.map(is_even, args))

  with get_executor(job_name='miniray_test_basic') as executor:
    results_miniray = list(executor.map(is_even, args))

  for a, b, c in zip(results_loop, results_threadpool, results_miniray, strict=True):
    assert a == b == c


def test_submit_result():
  with get_executor(job_name='miniray_test_result') as executor:
    future = executor.submit(is_even, 96)
    assert future.result() is True


@pytest.mark.parametrize("force_local", [True, False], ids=["local", "remote"])
def test_env_propagates_to_task_runtime(monkeypatch, force_local):
  key = "MINIRAY_TEST_LOCAL_ENV"
  value = "local_env_is_forwarded"
  monkeypatch.delenv(key, raising=False)
  timeout_seconds = 120

  with miniray.Executor(job_name='miniray_test_env',
                        priority=MINIRAY_PRIORITY,
                        queue_name=QUEUE_NAME,
                        limits={'memory': MINIRAY_MEMORY_GB},
                        env={key: value},
                        force_local=force_local) as executor:
    future = executor.submit(os.getenv, key)
    assert future.result(timeout=timeout_seconds) == value


def test_as_completed():
  with get_executor(job_name='miniray_test_as_completed') as executor:
    futures = [executor.submit(is_even, n) for n in range(10, 20)]
    results_completed = [future.result() for future in as_completed(futures)]
    assert sum(results_completed) == 5


def test_map_large_batches():
  with get_executor(job_name='miniray_test_map_100k') as executor:
    futures_100k = list(executor.fmap(is_even, range(100000), chunksize=1000))
    results_100k = miniray.log(futures_100k)
    assert len(results_100k) == 100000


def test_large_payloads():
  payload_mb = 100
  payload_bytes = payload_mb * 1024 * 1024
  num_tasks = 10

  with get_executor(job_name='miniray_test_100mb_payload') as executor:
    futures_payload = list(executor.fmap(make_random_payload, [payload_bytes] * num_tasks, chunksize=1))
    results_payload = miniray.log(futures_payload, desc="Receiving 100MB payloads")
    assert len(results_payload) == num_tasks
    total_bytes = sum(len(payload) for payload in results_payload)
    assert total_bytes == payload_bytes * num_tasks


def test_timeout():
  timeout_seconds = 1
  sleep_seconds = 2
  with miniray.Executor(job_name='miniray_test_timeout',
                        priority=MINIRAY_PRIORITY,
                        queue_name=QUEUE_NAME,
                        limits={'memory': MINIRAY_MEMORY_GB, 'timeout_seconds': timeout_seconds}) as executor:
    future = executor.submit(slow_sleep, sleep_seconds)
    with pytest.raises(miniray.MinirayError) as excinfo:
      future.result()
    assert excinfo.value.exception_type == "TimeoutError"


@pytest.mark.parametrize("hold_seconds", [30], ids=["dstate_30s"])
def test_d_state_timeout_reaping(hold_seconds):
  with get_executor(job_name=f"miniray_test_dstate_probe_{hold_seconds}s") as executor:
    if not executor.submit(remote_worker_supports_d_state_test).result(timeout=120):
      pytest.skip("remote worker does not support root loop mounts and fsfreeze")

  timeout_seconds = 10
  with miniray.Executor(job_name=f"miniray_test_dstate_{hold_seconds}s",
                        priority=MINIRAY_PRIORITY,
                        queue_name=QUEUE_NAME,
                        limits={"memory": MINIRAY_MEMORY_GB, "timeout_seconds": timeout_seconds}) as executor:
    future = executor.submit(block_in_frozen_filesystem, hold_seconds)
    t0 = time.monotonic()
    with pytest.raises(miniray.MinirayError) as excinfo:
      future.result(timeout=hold_seconds + 120)
    elapsed = time.monotonic() - t0

    assert excinfo.value.exception_type == "TimeoutError"
    assert elapsed >= hold_seconds - 2
    assert executor.submit(is_even, 96).result(timeout=60) is True


def test_class_method_submission():
    test1 = MinirayTestClass('test_value_1')
    test2 = MinirayTestClass('test_value_2')

    with get_executor(job_name='miniray_test_class_method') as executor:
      for obj, expected in [(test1, 'test_value_1_foo'), (test2, 'test_value_2_foo')]:
        future = executor.submit(obj.get_miniray_output, '_foo')
        assert future.result() == expected


def test_exception_propagation():
  with get_executor(job_name='miniray_test_exception_propagation') as executor:
    future = executor.submit(get_miniray_error)
    with pytest.raises(miniray.MinirayError) as excinfo:
      future.result()
    assert 'RuntimeError: Ruh roh!' in str(excinfo.value)


def test_memory_limit():
  memory_limit_gb = 0.5
  memory_limit_bytes = int(memory_limit_gb * 1024 * 1024 * 1024)

  under_limit_bytes = int(memory_limit_bytes * 0.9)
  over_limit_bytes = int(memory_limit_bytes * 1.1)

  # Allocate memory and return hash to avoid pickling overhead on return
  import hashlib
  def allocate_and_hash(size): return hashlib.md5(os.urandom(size)).hexdigest()

  with miniray.Executor(job_name='miniray_test_memory_limit',
                        priority=MINIRAY_PRIORITY,
                        queue_name=QUEUE_NAME,
                        limits={'memory': memory_limit_gb}) as executor:
    # 10% under the limit should pass
    future_under = executor.submit(allocate_and_hash, under_limit_bytes)
    assert len(future_under.result()) == 32  # md5 hex digest length

    # 10% over the limit should fail with OOM (SIGKILL)
    future_over = executor.submit(allocate_and_hash, over_limit_bytes)
    with pytest.raises(miniray.MinirayError) as excinfo:
      future_over.result()
    assert excinfo.value.exception_type == "ChildProcessError<-9>"


def test_early_shutdown():
  def it():
    for _ in range(100):
      yield from range(1000)
      time.sleep(0.1)

  with get_executor(job_name='miniray_test_early_shutdown_wait') as executor:
    futures = list(executor.fmap(is_even, range(10)))
    executor.shutdown(wait=True)
  assert all(f.done() for f in futures)  # all tasks completed
  assert all(isinstance(f.result(), bool) for f in futures)  # results are valid

  with get_executor(job_name='miniray_test_early_shutdown_cancel_read') as executor:
    futs = list(executor.fmap(is_even, range(10)))
    futs2 = list(executor.fmap(is_even, range(10, 20)))
    t0 = time.time()
    executor.shutdown(wait=True, cancel_futures=True)
  assert time.time() - t0 < 0.5
  assert all(f.done() for f in futs + futs2)  # all tasks completed
  assert any(f.cancelled() for f in futs + futs2)  # at least some were cancelled
  assert all(t and not t.is_alive() for t in executor._writer_threads + [executor._reader_thread])  # noqa: SLF001

  with get_executor(job_name='miniray_test_early_shutdown_cancel_submit') as executor:
    executor.fmap(is_even, it(), chunksize=1000)
    t0 = time.time()
    executor.shutdown(wait=True, cancel_futures=True)
  assert time.time() - t0 < 0.5
  assert all(t and not t.is_alive() for t in executor._writer_threads + [executor._reader_thread])  # noqa: SLF001


def test_zombie_processes():
  with get_executor(job_name='miniray_test_zombie') as executor:
    futures = [executor.submit(spawn_zombie) for _ in range(256)]
    results = [f.result() for f in futures]
  assert all(r == "done" for r in results)


def test_nonexistent_codedir():
  """Tasks submitted with a codedir that doesn't exist on the worker should fail, not crash the worker."""
  import tempfile
  tmpdir = tempfile.mkdtemp()
  executor = miniray.Executor(job_name='miniray_test_bad_codedir',
                              priority=MINIRAY_PRIORITY,
                              queue_name=QUEUE_NAME,
                              codedir=tmpdir,
                              limits={'memory': MINIRAY_MEMORY_GB})
  executor.__enter__()
  try:
    Path(tmpdir).rmdir()  # remove so it doesn't exist when the worker tries to use it
    future = executor.submit(is_even, 42)
    with pytest.raises(miniray.MinirayError):
      future.result(timeout=60)
  finally:
    executor.shutdown(cancel_futures=True)

  # verify the worker is still alive after handling the bad codedir
  with get_executor(job_name='miniray_test_bad_codedir_recovery') as executor:
    assert executor.submit(is_even, 96).result() is True
