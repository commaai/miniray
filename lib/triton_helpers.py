import os
import sys
import json
import time
import shutil
import signal
import subprocess
import urllib.request
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict
from redis import StrictRedis
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, wait_random
from tritonclient.http import InferenceServerClient
from tritonclient.utils import InferenceServerException

TRITON_REDIS_HOST = os.getenv('TRITON_REDIS_HOST', '127.0.0.1')
TRITON_SERVER_ADDRESS = os.getenv('TRITON_SERVER_ADDRESS', '127.0.0.1:8000')
TRITON_SHM_DIR = Path('/dev/shm')
TRITON_MODEL_REPOSITORY = Path(os.getenv('TRITON_MODEL_REPOSITORY', '/dev/shm/model-repository'))
TRITON_MODEL_STALE_AFTER_SECONDS_PARAMETER = 'stale_after_seconds'

IOConfig = TypedDict('IOConfig', {'name': str, 'data_type': str, 'dims': list[int]})
ModelConfig = TypedDict('ModelConfig', {'input': list[IOConfig], 'output': list[IOConfig]})

def _check_triton_server_health(url: str, timeout: int = 3, scheme: str = "http") -> None:
  if "://" not in url:
    url = f"{scheme}://{url}"
  urllib.request.urlopen(f"{url}/v2/health/live", timeout=timeout)

def _is_model_loading(client: InferenceServerClient, model_name: str, model_version: str):
  repo_index = client.get_model_repository_index()
  for entry in repo_index:
    if entry.get("name") == model_name and entry.get("version") == model_version and entry.get("state") == "LOADING":
      return True
  return False

check_triton_server_health = retry(
  stop=stop_after_delay(15),
  wait=wait_fixed(1),
  reraise=True,
)(_check_triton_server_health)
wait_for_triton_server = retry(
  stop=stop_after_delay(60),
  wait=wait_fixed(2),
  reraise=True,
)(_check_triton_server_health)

@retry(stop=stop_after_attempt(3), wait=wait_random(1, 2), reraise=True)
def get_triton_inference_stats(client: InferenceServerClient):
  return client.get_inference_statistics()['model_stats']

@retry(stop=stop_after_attempt(3), wait=wait_random(1, 2), reraise=True)
def load_triton_model(client: InferenceServerClient, model: str, config: ModelConfig,
                      load_timeout = 60, model_version: str = '1'):
  if _is_model_loading(client, model, model_version):
    deadline = time.perf_counter() + load_timeout
    while time.perf_counter() < deadline and _is_model_loading(client, model, model_version):
      time.sleep(min(5, load_timeout))
    assert client.is_model_ready(model, model_version)
    return
  return client.load_model(model, config=json.dumps(config))

def cleanup_triton_model_versions(client: InferenceServerClient, model: str, unload_timeout: float = 60):
  deadline = time.monotonic() + unload_timeout
  while True:
    versions = [entry for entry in client.get_model_repository_index() if entry['name'] == model]
    if not any(entry.get('state') == 'UNLOADING' for entry in versions):
      break
    if time.monotonic() >= deadline:
      raise TimeoutError(f'Triton model {model} did not finish unloading within {unload_timeout} seconds: {versions}')
    time.sleep(0.1)
  active = {entry['version'] for entry in versions if entry.get('state') in ('READY', 'LOADING')}
  for version_dir in (TRITON_MODEL_REPOSITORY / model).glob('*'):
    if version_dir.is_dir() and version_dir.name.isdigit() and version_dir.name not in active:
      shutil.rmtree(version_dir)

def setup_triton_model(func: Callable[..., ModelConfig]):
  @wraps(func)
  def wrapper(*self: Any,
    client: InferenceServerClient,
    model: str,
    redis: Optional[StrictRedis] = None,
    load_timeout = 60,
    model_version: str = '1') -> None:
      model_dir = TRITON_MODEL_REPOSITORY / model / model_version
      keep_count = int(os.getenv('TRITON_MAX_CHECKPOINTS_PER_EID', '1'))
      if keep_count < 1:
        raise ValueError('TRITON_MAX_CHECKPOINTS_PER_EID must be at least 1')
      version_policy = {'latest': {'num_versions': keep_count}}
      if redis is None:
        redis = StrictRedis(host=TRITON_REDIS_HOST, port=6379, db=8)
      with redis.lock(model, timeout=10*60):
        ready = client.is_model_ready(model, model_version)
        if ready:
          config = client.get_model_config(model, model_version)
        else:
          cleanup_triton_model_versions(client, model)
          shutil.rmtree(model_dir, ignore_errors=True)
          model_dir.mkdir(parents=True, exist_ok=True)
          config = func(*self, model_dir)
        if not ready or config.get('version_policy') != version_policy:
          config['version_policy'] = version_policy
          load_triton_model(client, model, config, load_timeout=load_timeout, model_version=model_version)
        cleanup_triton_model_versions(client, model)
        if not client.is_model_ready(model, model_version):
          raise RuntimeError(f'Triton model {model} version {model_version} is outside the latest-{keep_count} window')
        mtime = time.time()
        try: os.utime(model_dir, (mtime, mtime))
        except OSError: pass
  return wrapper

def unload_triton_model(client: InferenceServerClient, model: str):
  client.unload_model(model)
  cleanup_triton_model_versions(client, model)
  try: shutil.rmtree(TRITON_MODEL_REPOSITORY / model)
  except FileNotFoundError: pass
  for f in Path("/dev/shm").glob(f"{model}_*.parameters"):
    f.unlink(missing_ok=True)

def unload_triton_models(client: InferenceServerClient, model: Optional[str] = None):
  for name in {stats['name'] for stats in get_triton_inference_stats(client)}:
    if model is None or model == name:
      print(f"Unloading {name}")
      unload_triton_model(client, name)

  if model is None:
    for subdir in TRITON_MODEL_REPOSITORY.iterdir():
      print("Removing leftover model data:", subdir)
      try: shutil.rmtree(subdir)
      except FileNotFoundError: pass

# NOTE: This function must be run as the root user or it will throw a PermissionError
def kill_triton_processes_by_name(name: str) -> None:
  container_id = get_triton_container_id()
  output = subprocess.check_output(["docker", "top", container_id, "-eo", "pid,args"], text=True, timeout=5)
  for line in output.splitlines()[1:]:
    fields = line.strip().split(maxsplit=1)
    if len(fields) != 2: continue
    pid_text, command = fields
    if name not in command: continue
    try: os.kill(int(pid_text), signal.SIGKILL)
    except ProcessLookupError: pass

# NOTE: This function must also run as root, since the triton_python_backend_shm_region files are
# created directly by the triton server
def unlink_triton_shm_files() -> None:
  for f in TRITON_SHM_DIR.glob("triton_*_backend_shm_region_*"):
    f.unlink(missing_ok=True)

def get_triton_container_id() -> str:
  container_ids = subprocess.check_output(
    ["docker", "ps", "--format", "{{.ID}}", "--filter", "name=tritonserver"]).decode('utf-8').strip()
  if not container_ids:
    raise RuntimeError("No tritonserver container found")
  return container_ids.split('\n')[0]

def cleanup_triton(client: InferenceServerClient) -> None:
  kill_triton_processes_by_name("VLLM::EngineCore")
  unload_triton_models(client)
  kill_triton_processes_by_name("triton_python_backend_stub")
  unlink_triton_shm_files()

def unload_stale_models(triton_client: InferenceServerClient, redis_client: StrictRedis, keep_model_name: str) -> None:
  last_used = {}
  for model in get_triton_inference_stats(triton_client):
    last_inference_time = model['last_inference']//1000
    try: model_mtime = (TRITON_MODEL_REPOSITORY / model['name'] / model['version']).stat().st_mtime
    except FileNotFoundError: model_mtime = 0
    last_used[model['name']] = max(last_used.get(model['name'], 0), last_inference_time, model_mtime)
  for name, last_inference_time in last_used.items():
    if name == keep_model_name:
      continue
    try: parameters = triton_client.get_model_config(name).get('parameters', {})
    except InferenceServerException: continue
    model_stale_after_seconds = float(
      parameters.get(TRITON_MODEL_STALE_AFTER_SECONDS_PARAMETER, {}).get('string_value', 30*60))
    if time.time() - last_inference_time > model_stale_after_seconds:
      with redis_client.lock(name, timeout=10*60):
        unload_triton_model(triton_client, name)

if __name__ == '__main__':
  import argparse
  import json

  default_host, default_port = TRITON_SERVER_ADDRESS.split(':')
  parser = argparse.ArgumentParser(description='Triton model utilities')
  subparsers = parser.add_subparsers(dest='command', help='Available commands')

  logs_parser = subparsers.add_parser('logs', help='Show triton server logs (docker)')
  shell_parser = subparsers.add_parser('shell', help='Open a bash shell in the triton server container (docker)')
  list_parser = subparsers.add_parser('list', help='List models loaded in triton')
  stats_parser = subparsers.add_parser('stats', help='Get triton inference server statistics')
  unload_parser = subparsers.add_parser('unload', help='Unload triton models')
  unload_parser.add_argument('model', nargs="?", help='Model name to unload')

  for p in [list_parser, stats_parser, unload_parser]:
    p.add_argument('host', nargs='?', default=default_host, help='hostname of the inference server')
    p.add_argument('-p', '--port', type=int, default=default_port, help='port number of the inference server')

  args, _ = parser.parse_known_args()  # triton logs passes the args through to the `docker logs` command
  if args.command == 'logs':
    os.execvp("docker", ["docker", "logs", *sys.argv[2:], get_triton_container_id()])

  args, _ = parser.parse_known_args()
  if args.command == 'shell':
    os.execvp("docker", ["docker", "exec", "-it", *sys.argv[2:], get_triton_container_id(), "/bin/bash"])

  triton_client = InferenceServerClient(url=f'{args.host}:{args.port}', verbose=False)
  if args.command == 'unload':
    unload_triton_models(triton_client, model=args.model)
  elif args.command in ('list', 'stats'):
    inference_stats = triton_client.get_inference_statistics()['model_stats']
    if not inference_stats:
      print("No models loaded")
    elif args.command == 'list':
      print(f"Models loaded in triton server at {args.host}:{args.port}:")
      for stat in inference_stats:
        print(f"  {stat['name']}")
    else:
      print(f"-- Inference statistics for triton server at {args.host}:{args.port} --")
      for stat in inference_stats:
        print(json.dumps(stat, indent=2))
        print()
