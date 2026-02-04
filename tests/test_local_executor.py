"""Tests for LocalExecutor that don't require Redis or distributed workers."""
import os
import time
import numpy as np
import pytest

from concurrent.futures import as_completed

# Force local execution for these tests
os.environ['MINIRAY_FORCE_LOCAL'] = '1'

import miniray


class MinirayTestClass:
  def __init__(self, value):
    self.value = value
  def get_miniray_output(self, x):
    return self.value + x


def get_error():
  raise RuntimeError("Test error!")


def is_even(n):
  return n % 2 == 0


def make_random_payload(size: int) -> bytes:
  return os.urandom(size)


def slow_task(seconds: float) -> str:
  time.sleep(seconds)
  return "done"


def test_local_map():
  """Test basic map operation with local executor."""
  args = list(range(100))
  expected = [is_even(n) for n in args]

  with miniray.Executor(job_name='test_local_map') as executor:
    results = list(executor.map(is_even, args))

  assert results == expected


def test_local_submit():
  """Test submit operation with local executor."""
  with miniray.Executor(job_name='test_local_submit') as executor:
    future = executor.submit(is_even, 42)
    assert future.result() is True

    future = executor.submit(is_even, 43)
    assert future.result() is False


def test_local_as_completed():
  """Test as_completed with local executor."""
  with miniray.Executor(job_name='test_local_as_completed') as executor:
    futures = [executor.submit(is_even, n) for n in range(10)]
    results = [f.result() for f in as_completed(futures)]
    assert sum(results) == 5  # 0, 2, 4, 6, 8 are even


def test_local_fmap():
  """Test fmap (future map) with local executor."""
  with miniray.Executor(job_name='test_local_fmap') as executor:
    futures = list(executor.fmap(is_even, range(50), chunksize=10))
    results = miniray.log(futures, desc="Local fmap test")
    assert len(results) == 50
    assert sum(results) == 25  # half are even


def test_local_class_method():
  """Test class method serialization with local executor."""
  obj = MinirayTestClass('value_')

  with miniray.Executor(job_name='test_local_class_method') as executor:
    future = executor.submit(obj.get_miniray_output, 'suffix')
    assert future.result() == 'value_suffix'


def test_local_exception():
  """Test exception propagation with local executor."""
  with miniray.Executor(job_name='test_local_exception') as executor:
    future = executor.submit(get_error)
    with pytest.raises(Exception):
      future.result()


def test_local_payload():
  """Test payload handling with local executor."""
  payload_size = 1024 * 1024  # 1 MB

  with miniray.Executor(job_name='test_local_payload') as executor:
    future = executor.submit(make_random_payload, payload_size)
    result = future.result()
    assert len(result) == payload_size


def test_local_executor_direct():
  """Test LocalExecutor directly."""
  with miniray.LocalExecutor() as executor:
    futures = [executor.submit(is_even, n) for n in range(20)]
    results = [f.result() for f in futures]
    assert sum(results) == 10
