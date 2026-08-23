from __future__ import annotations

import shutil
from typing import Union
from pathlib import Path
import subprocess
import pwd
import os
import tempfile
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
  # Install packages into the venv so runtime imports are self-contained.
  sync_cmd = ['uv', 'sync', '--project', codedir, '--frozen', '--no-editable', '--link-mode', 'clone']

  errs = []
  for i in range(N_RETRIES):
    # Staged source is read-only; a private build root also isolates concurrent syncs.
    with tempfile.TemporaryDirectory(prefix='miniray-setuptools-', dir='/tmp') as build_root:
      os.chown(build_root, user_id, -1)
      setuptools_config = Path(build_root) / 'setuptools.cfg'
      # Redirect all Setuptools output and clean it between local package builds.
      setuptools_config.write_text(
        f'[aliases]\nbdist_wheel = clean --all bdist_wheel\n\n'
        f'[egg_info]\negg_base = {build_root}\n\n'
        f'[build]\nbuild_base = {build_root}/build\n')
      os.chown(setuptools_config, user_id, -1)

      try:
        subprocess.run(
          sync_cmd, env={
            **os.environ,
            'DIST_EXTRA_CONFIG': str(setuptools_config),
            'UV_PROJECT_ENVIRONMENT': str(venv_dir),
            # These builds share build_base, so uv must run them serially.
            'UV_CONCURRENT_BUILDS': '1',
          },
          user=user_id, check=True, capture_output=True)
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
