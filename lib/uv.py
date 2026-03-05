import shutil
from typing import Union
from pathlib import Path
import subprocess
import pwd
import os

N_RETRIES = 5


def _parse_uv_sync_stderr(stderr):
  if stderr is None: return ''
  stderr = stderr.decode('utf-8')
  errs = [line for line in stderr.split('\n') if line.startswith('error')] # filter out infos and warnings
  return '\n'.join(errs) if len(errs) else stderr

def _base_venv_path(user_id: int):
  return Path(pwd.getpwuid(user_id).pw_dir) / ".job_venvs"

def _start_uv_sync(codedir: Union[str, Path], user_id: int, venv_name: str) -> subprocess.Popen:
  venv_dir = _base_venv_path(user_id) / venv_name
  sync_cmd = ['uv', 'sync', '--project', str(codedir), '--frozen']
  if os.getenv('CI'):
    sync_cmd += ['--link-mode', 'symlink'] # hardlinking is slow in docker
  return subprocess.Popen(sync_cmd, env={**os.environ, 'UV_PROJECT_ENVIRONMENT': str(venv_dir)},
                          user=user_id, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def _cleanup_venvs(user_id: int, keep_venvs: list[str]):
  base_dir = _base_venv_path(user_id)
  if not base_dir.exists():
    return
  for venv in base_dir.iterdir():
    if venv.name not in keep_venvs:
      shutil.rmtree(venv)


class VenvManager:
  """Manages non-blocking venv syncs with retry logic."""

  def __init__(self, user_id: int, cache_size: int):
    from lru import LRU
    self.user_id = user_id
    self._cache: dict[str, str] = LRU(cache_size)
    self._pending: dict[str, tuple[subprocess.Popen, str, int]] = {}  # job -> (proc, codedir, retries)

  def __contains__(self, job: str) -> bool:
    return job in self._cache

  def __getitem__(self, job: str) -> str:
    return self._cache[job]

  def sync(self, job: str, codedir: str):
    """Start a venv sync for a job if not already cached or in progress."""
    if job in self._cache or job in self._pending:
      return
    if not Path(codedir).exists():
      return
    self._pending[job] = (_start_uv_sync(codedir, self.user_id, job), codedir, 0)
    print(f"[worker] starting venv sync for {job}")

  def poll(self):
    """Check pending syncs. Move completed ones to cache, retry failures."""
    for job in list(self._pending):
      proc, codedir, retries = self._pending[job]
      if proc.poll() is None:
        continue
      if proc.returncode == 0:
        self._cache[job] = str(_base_venv_path(self.user_id) / job)
        del self._pending[job]
        _cleanup_venvs(self.user_id, keep_venvs=list(self._cache.keys()))
        print(f"[worker] venv ready for {job}")
      else:
        stderr = _parse_uv_sync_stderr(proc.stderr.read() if proc.stderr else None)
        retries += 1
        if retries >= N_RETRIES:
          print(f"[worker] venv sync failed for {job} after {N_RETRIES} retries: {stderr}")
          del self._pending[job]
        else:
          if retries >= 3:
            shutil.rmtree(_base_venv_path(self.user_id) / job, ignore_errors=True)
          print(f"[worker] venv sync retry {retries}/{N_RETRIES} for {job}: {stderr}")
          self._pending[job] = (_start_uv_sync(codedir, self.user_id, job), codedir, retries)
