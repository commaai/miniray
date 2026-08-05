import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

from miniray.lib import uv


def write_setuptools_project(project: Path, name: str, setup_py: str | None = None):
  package = project / name
  package.mkdir(parents=True)
  (package / '__init__.py').write_text(f'NAME = {name!r}\n')
  (project / 'pyproject.toml').write_text(f'''\
[build-system]
requires = ["setuptools==81.0.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.0.0"
requires-python = ">=3.12"

[tool.setuptools]
packages = ["{name}"]
''')
  if setup_py is not None:
    (project / 'setup.py').write_text(setup_py)

  lock_env = {
    **os.environ,
    'UV_CACHE_DIR': str(project.parent / f'{project.name}-lock-cache'),
    'UV_PYTHON_DOWNLOADS': 'never',
  }
  subprocess.run(['uv', 'lock', '--project', project], env=lock_env, check=True, capture_output=True)


def configure_sync(tmp_path: Path, monkeypatch):
  monkeypatch.setenv('UV_CACHE_DIR', str(tmp_path / 'sync-cache'))
  monkeypatch.setenv('UV_PYTHON_DOWNLOADS', 'never')
  monkeypatch.setattr(uv, 'base_venv_path', lambda _user_id: tmp_path / 'venvs')
  monkeypatch.setattr(uv, 'N_RETRIES', 1)


def source_snapshot(source: Path):
  return {
    path.relative_to(source): None if path.is_dir() else path.read_bytes()
    for path in source.rglob('*')
  }


def set_source_read_only(source: Path, read_only: bool):
  for path in [source, *source.rglob('*')]:
    if path.is_dir():
      path.chmod(0o500 if read_only else 0o700)
    else:
      path.chmod(0o400 if read_only else 0o600)


def test_sync_venv_cache_is_local(tmp_path, monkeypatch):
  project = tmp_path / 'locality'
  write_setuptools_project(project, 'local_package')
  configure_sync(tmp_path, monkeypatch)

  before = source_snapshot(project)
  egg_base = tmp_path / 'egg-info'
  egg_base.mkdir()
  setuptools_config = tmp_path / 'setuptools.cfg'
  setuptools_config.write_text(f'[egg_info]\negg_base = {egg_base}\n')
  monkeypatch.setenv('DIST_EXTRA_CONFIG', str(setuptools_config))
  set_source_read_only(project, True)

  try:
    venv = uv.sync_venv_cache(project, os.getuid(), 'locality')
  finally:
    set_source_read_only(project, False)

  # Locality: a non-editable import must resolve inside the job venv.
  origin = Path(subprocess.check_output(
    [venv / 'bin' / 'python', '-c', 'import local_package; print(local_package.__file__)'],
    env={**os.environ, 'PYTHONPATH': ''}, text=True,
  ).strip())
  assert origin.is_relative_to(venv), f'{origin} is outside {venv}'

  # Source immutability: syncing must not create or modify files in staged code.
  assert source_snapshot(project) == before


COUNTING_SETUP = '''\
import os
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py

class CountingBuildPy(build_py):
  def run(self):
    with Path(os.environ["MINIRAY_TEST_BUILD_COUNTER"]).open("a") as counter:
      counter.write("build\\n")
    super().run()

setup(cmdclass={"build_py": CountingBuildPy})
'''


def test_sync_venv_cache_reuses_warm_cache(tmp_path, monkeypatch):
  project = tmp_path / 'warm-cache'
  write_setuptools_project(project, 'cached_package', COUNTING_SETUP)
  configure_sync(tmp_path, monkeypatch)
  counter = tmp_path / 'build-count'
  monkeypatch.setenv('MINIRAY_TEST_BUILD_COUNTER', str(counter))

  uv.sync_venv_cache(project, os.getuid(), 'cold')
  # Cold-build baseline: the backend ran to populate uv's wheel cache.
  cold_builds = counter.read_bytes()
  assert cold_builds

  uv.sync_venv_cache(project, os.getuid(), 'warm')
  # Warm-cache hit: a different venv installs the cached wheel without rebuilding it.
  assert counter.read_bytes() == cold_builds


def barrier_setup(name: str, other: str) -> str:
  return f'NAME = {name!r}\nOTHER = {other!r}\n' + '''\
import os
import time
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py

def wait_for(path):
  deadline = time.monotonic() + 15
  while not path.exists():
    if time.monotonic() >= deadline:
      raise RuntimeError("timed out waiting for " + str(path))
    time.sleep(0.01)

class BarrierBuildPy(build_py):
  def run(self):
    barrier = Path(os.environ["MINIRAY_TEST_BUILD_BARRIER"])
    (barrier / (NAME + ".ready")).touch()
    wait_for(barrier / (OTHER + ".ready"))
    if NAME == "fixture_a":
      super().run()
      (barrier / (NAME + ".built")).touch()
      wait_for(barrier / (OTHER + ".built"))
    else:
      wait_for(barrier / (OTHER + ".built"))
      super().run()
      (barrier / (NAME + ".built")).touch()

setup(cmdclass={"build_py": BarrierBuildPy})
'''


def test_sync_venv_cache_concurrent_builds_are_isolated(tmp_path, monkeypatch):
  names = ('fixture_a', 'fixture_b')
  projects = []
  for name, other in zip(names, reversed(names), strict=True):
    project = tmp_path / name
    write_setuptools_project(project, name, barrier_setup(name, other))
    projects.append(project)

  configure_sync(tmp_path, monkeypatch)
  barrier = tmp_path / 'build-barrier'
  barrier.mkdir()
  monkeypatch.setenv('MINIRAY_TEST_BUILD_BARRIER', str(barrier))

  # Concurrency: the backend barrier forces both uv sync processes into wheel builds simultaneously.
  with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [
      executor.submit(uv.sync_venv_cache, project, os.getuid(), name)
      for project, name in zip(projects, names, strict=True)
    ]
    venvs = [future.result(timeout=30) for future in futures]
  assert {path.name for path in barrier.iterdir()} == {
    'fixture_a.ready', 'fixture_a.built', 'fixture_b.ready', 'fixture_b.built',
  }

  # Concurrent-build isolation: neither wheel may absorb files from the other build directory.
  for name, other, venv in zip(names, reversed(names), venvs, strict=True):
    site_packages = next((venv / 'lib').glob('python*/site-packages'))
    assert (site_packages / name / '__init__.py').is_file()
    assert not (site_packages / other).exists()
