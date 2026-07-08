import os
import time
from pathlib import Path
import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

import miniray
from .dstate_helpers import (
  block_in_frozen_filesystem,
  wait_for_worker_to_disappear,
)

MINIRAY_PRIORITY = 1000
MINIRAY_MEMORY_GB = 0.4
DSTATE_TASK_COUNT = 4
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


@pytest.mark.dstate
def test_d_state_tasks_do_not_crash_worker():
  hold_seconds = 30
  timeout_seconds = 10

  with miniray.Executor(job_name="miniray_test_dstate_no_crash",
                        priority=MINIRAY_PRIORITY,
                        queue_name=QUEUE_NAME,
                        limits={"memory": MINIRAY_MEMORY_GB, "timeout_seconds": timeout_seconds}) as executor:
    futures = [executor.submit(block_in_frozen_filesystem, hold_seconds) for _ in range(DSTATE_TASK_COUNT)]
    t0 = time.monotonic()
    errors = []
    for future in as_completed(futures, timeout=hold_seconds + 120):
      with pytest.raises(miniray.MinirayError) as excinfo:
        future.result()
      errors.append(excinfo.value)
    elapsed = time.monotonic() - t0

    assert len(errors) == DSTATE_TASK_COUNT
    assert {error.exception_type for error in errors} == {"TimeoutError"}
    assert elapsed >= hold_seconds - 2
    assert executor.submit(is_even, 96).result(timeout=60) is True


@pytest.mark.dstate
def test_d_state_task_crashes_worker():
  hold_seconds = 90
  timeout_seconds = 10

  with miniray.Executor(job_name="miniray_test_dstate_crash",
                        priority=MINIRAY_PRIORITY,
                        queue_name=QUEUE_NAME,
                        limits={"memory": MINIRAY_MEMORY_GB, "timeout_seconds": timeout_seconds}) as executor:
    future = executor.submit(block_in_frozen_filesystem, hold_seconds)
    with pytest.raises(miniray.MinirayError) as excinfo:
      future.result(timeout=hold_seconds + 120)

  assert excinfo.value.exception_type == "RuntimeError"
  assert "task lost" in excinfo.value.exception_desc
  wait_for_worker_to_disappear(QUEUE_NAME, excinfo.value.worker)


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


@pytest.mark.dstate
def test_malformed_job_metadata_does_not_crash_worker():
  from redis import StrictRedis
  from miniray.executor import get_metadata_key

  with miniray.Executor(job_name='miniray_test_malformed_metadata',
                        priority=MINIRAY_PRIORITY, queue_name=QUEUE_NAME,
                        limits={'memory': MINIRAY_MEMORY_GB},
                        queue_timeout=60) as ex:
    r = StrictRedis(host=ex.config.redis_host, port=6379, db=1)
    r.set(get_metadata_key(ex.submit_queue_id), b'not valid json')

    future = ex.submit(is_even, 42)
    with pytest.raises(miniray.MinirayError):
      future.result(timeout=90)

  with get_executor(job_name='miniray_test_malformed_metadata_recovery') as executor:
    assert executor.submit(is_even, 96).result(timeout=60) is True


@pytest.mark.dstate
def test_brief_redis_crash_does_not_crash_worker():
  from redis import StrictRedis

  redis_host = os.environ.get('REDIS_HOST', 'redis.comma.internal')
  r = StrictRedis(host=redis_host, port=6379, db=1)

  with get_executor(job_name='miniray_test_redis_crash') as ex:
    assert ex.submit(is_even, 96).result(timeout=60) is True

  try:
    r.shutdown()
  except Exception:
    pass

  deadline = time.monotonic() + 30
  while time.monotonic() < deadline:
    try:
      r.ping()
      break
    except Exception:
      time.sleep(0.5)
  else:
    pytest.fail("redis did not come back up")

  with get_executor(job_name='miniray_test_redis_crash_recovery') as ex:
    assert ex.submit(is_even, 96).result(timeout=60) is True


@pytest.mark.dstate
def test_more_jobs_than_cache_size_does_not_crash_worker():
  from miniray.lib.helpers import JOB_CACHE_SIZE

  executors = []
  try:
    for i in range(JOB_CACHE_SIZE * 2):
      ex = get_executor(job_name=f'miniray_test_job_overflow_{i}')
      ex.__enter__()
      executors.append(ex)
      ex.submit(is_even, 96)
      ex.submit(is_even, 96)

    check_future = executors[0].submit(is_even, 96)
    assert check_future.result(timeout=60) is True
  finally:
    for ex in executors:
      ex.shutdown(wait=False, cancel_futures=True)
