import shutil
from typing import Union
from pathlib import Path
import subprocess
import pwd
import os

N_RETRIES = 5


def parse_uv_sync_stderr(stderr):
  if stderr is None:
    return ''
  errs = [err for err in stderr.decode('utf-8').split('\n') if err.startswith('error')] # filter out infos and warnings
  err = "\n".join(errs)
  return err

def base_venv_path(user_id: int):
  return Path(pwd.getpwuid(user_id).pw_dir) / ".job_venvs"

def sync_venv_cache(codedir: Union[str, Path], user_id: int, venv_name: str):
  venv_dir = base_venv_path(user_id) / venv_name
  sync_cmd = ['uv', 'sync', '--project', codedir, '--frozen']
  if os.getenv('CI'):
    sync_cmd += ['--link-mode', 'symlink'] # hardlinking is slow in docker

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
