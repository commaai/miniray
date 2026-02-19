import os
import time
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
    assert excinfo.value.exception_type == "ChildProcessError<9>"


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
  assert all(t and not t.is_alive() for t in executor._writer_threads + [executor._reader_thread])

  with get_executor(job_name='miniray_test_early_shutdown_cancel_submit') as executor:
    executor.fmap(is_even, it(), chunksize=1000)
    t0 = time.time()
    executor.shutdown(wait=True, cancel_futures=True)
  assert time.time() - t0 < 0.5
  assert all(t and not t.is_alive() for t in executor._writer_threads + [executor._reader_thread])
