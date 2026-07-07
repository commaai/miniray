from __future__ import annotations

import shutil
from typing import Union
from pathlib import Path
import subprocess
import pwd
import os
from lru import LRU

N_RETRIES = 5


def parse_uv_sync_stderr(stderr):
  if stderr is None: return ''
  stderr = stderr.decode('utf-8')
  errs = [line for line in stderr.split('\n') if line.startswith('error')] # filter out infos and warnings
  return '\n'.join(errs) if len(errs) else stderr

def base_venv_path(user_id: int):
  return Path(pwd.getpwuid(user_id).pw_dir) / ".job_venvs"

def pycache_dir_for_venv(venv_name: str, user_id: int) -> Path:
  return Path(f"/var/cache/miniray/pycache_{user_id}") / venv_name

def sync_venv_cache(codedir: Union[str, Path], user_id: int, venv_name: str):
  venv_dir = base_venv_path(user_id) / venv_name
  # TODO: Try hardlink mode once the uv cache and job venvs share one writable mount.
  sync_cmd = ['uv', 'sync', '--project', codedir, '--frozen', '--link-mode', 'symlink']

  errs = []
  for i in range(N_RETRIES):
    try:
      subprocess.run(sync_cmd, env={**os.environ, 'UV_PROJECT_ENVIRONMENT': str(venv_dir)}, user=user_id, check=True, capture_output=True)
      return venv_dir
    except subprocess.CalledProcessError as e:
      errs.append(f'try {i}: {parse_uv_sync_stderr(e.stderr)}')
      if i >=3:
        try:
          shutil.rmtree(venv_dir)
        except Exception:
          pass
  raise ValueError(f"Failed syncing venv={venv_dir} to {codedir} {N_RETRIES} times \n" + "\n".join(errs))


def cleanup_venvs(user_id: int, keep_venvs: list[str]):
  base_dir = base_venv_path(user_id)
  if not base_dir.exists():
    return

  for venv in base_dir.iterdir():
    if venv.name not in keep_venvs:
      shutil.rmtree(venv)
      shutil.rmtree(pycache_dir_for_venv(venv.name, user_id), ignore_errors=True)


def populate_venv_cache_from_disk(venv_cache: LRU[str, str], user_id: int) -> None:
  base_dir = base_venv_path(user_id)
  if not base_dir.exists():
    return
  entries = sorted(base_dir.iterdir(), key=lambda p: p.stat().st_mtime)
  for entry in entries[-venv_cache.get_size():]:
    venv_cache[entry.name] = str(entry)
