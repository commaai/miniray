import urllib.error
import pytest

from miniray.lib import triton_helpers


@pytest.mark.parametrize("error", [TimeoutError(), ConnectionResetError()])
def test_triton_health_monitor_tolerates_transient_failures(monkeypatch, error):
  def fail_health_check(url):
    raise error

  times = iter([100.0, 189.9, 190.0])
  monitor = triton_helpers.TritonHealthMonitor(90, clock=lambda: next(times))
  monkeypatch.setattr(triton_helpers, "_check_triton_server_health", fail_health_check)

  monitor.check("localhost:8000")
  monitor.check("localhost:8000")
  with pytest.raises(type(error)):
    monitor.check("localhost:8000")


def test_triton_health_monitor_resets_after_recovery(monkeypatch):
  outcomes = iter([TimeoutError(), None, TimeoutError()])

  def health_check(url):
    if (error := next(outcomes)) is not None:
      raise error

  times = iter([100.0, 1000.0])
  monitor = triton_helpers.TritonHealthMonitor(90, clock=lambda: next(times))
  monkeypatch.setattr(triton_helpers, "_check_triton_server_health", health_check)

  monitor.check("localhost:8000")
  monitor.check("localhost:8000")
  monitor.check("localhost:8000")


def test_triton_health_monitor_does_not_hide_connection_refusal(monkeypatch):
  error = urllib.error.URLError(ConnectionRefusedError())

  def fail_health_check(url):
    raise error

  monitor = triton_helpers.TritonHealthMonitor(90)
  monkeypatch.setattr(triton_helpers, "_check_triton_server_health", fail_health_check)

  with pytest.raises(urllib.error.URLError):
    monitor.check("localhost:8000")
