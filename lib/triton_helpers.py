import os
import re
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
TRITON_CHECKPOINT_PATTERN = re.compile(r'([\da-fA-F]{8}(?:-[\da-fA-F]{4}){3}-[\da-fA-F]{12})_(-1|[0-9]+)_.+')

IOConfig = TypedDict('IOConfig', {'name': str, 'data_type': str, 'dims': list[int]})
ModelConfig = TypedDict('ModelConfig', {'input': list[IOConfig], 'output': list[IOConfig]})

def _check_triton_server_health(url: str, timeout: int = 3, scheme: str = "http") -> None:
  if "://" not in url:
    url = f"{scheme}://{url}"
  urllib.request.urlopen(f"{url}/v2/health/live", timeout=timeout)

def _is_model_loading(client: InferenceServerClient, model_name: str):
  repo_index = client.get_model_repository_index()
  for entry in repo_index:
    if entry.get("name", "") == model_name and entry.get("state", "") == "LOADING":
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
def load_triton_model(client: InferenceServerClient, model: str, config: ModelConfig, load_timeout = 60):
  if _is_model_loading(client, model):
    deadline = time.perf_counter() + load_timeout
    while time.perf_counter() < deadline and _is_model_loading(client, model):
      time.sleep(min(5, load_timeout))
    assert client.is_model_ready(model)
    return
  return client.load_model(model, config=json.dumps(config))

def retain_recent_checkpoints(client: InferenceServerClient, redis: StrictRedis, model_name: str):
  """Evict older epochs; the caller holds the EID lock through eviction and loading."""
  match = TRITON_CHECKPOINT_PATTERN.fullmatch(model_name)
  if match is None:
    return
  eid, epoch = match[1].lower(), int(match[2])
  keep_count = int(os.getenv('TRITON_MAX_CHECKPOINTS_PER_EID', '1'))
  if keep_count < 1:
    raise ValueError('TRITON_MAX_CHECKPOINTS_PER_EID must be at least 1')
  checkpoints = {}
  for model in client.get_model_repository_index():
    other = TRITON_CHECKPOINT_PATTERN.fullmatch(model['name'])
    if other and other[1].lower() == eid and model.get('state') in ('READY', 'LOADING', 'UNLOADING'):
      checkpoints[model['name']] = int(other[2])
  retained = sorted({epoch, *checkpoints.values()}, reverse=True)[:keep_count]
  for name, other_epoch in checkpoints.items():
    if other_epoch not in retained:
      with redis.lock(name, timeout=10*60):
        unload_triton_model(client, name)
  if epoch not in retained:
    raise RuntimeError(f'Checkpoint {eid}/{epoch} is outside the Triton retention window: {retained}')

def setup_triton_model(func: Callable[..., ModelConfig]):
  @wraps(func)
  def wrapper(*self: Any,
    client: InferenceServerClient,
    model: str,
    redis: Optional[StrictRedis] = None,
    load_timeout = 60) -> None:
      model_dir = TRITON_MODEL_REPOSITORY / model / '1'
      if redis is None:
        redis = StrictRedis(host=TRITON_REDIS_HOST, port=6379, db=8)
      checkpoint = TRITON_CHECKPOINT_PATTERN.fullmatch(model)
      model_group = checkpoint[1].lower() if checkpoint else model
      with redis.lock(f'triton-checkpoints/{model_group}', timeout=10*60):
        retain_recent_checkpoints(client, redis, model)
        with redis.lock(model, timeout=10*60):
          if client.is_model_ready(model):  # if already loaded, bump the mtime and return
            mtime = time.time()
            try: os.utime(model_dir, (mtime, mtime))
            except OSError: pass
            return
          shutil.rmtree(model_dir, ignore_errors=True)
          model_dir.mkdir(parents=True, exist_ok=True)
          config = func(*self, model_dir)
          load_triton_model(client, model, config, load_timeout=load_timeout)
          assert client.is_model_ready(model)
  return wrapper

def unload_triton_model(client: InferenceServerClient, model: str, unload_timeout: float = 60):
  client.unload_model(model)
  deadline = time.monotonic() + unload_timeout
  while any(entry['name'] == model and entry.get('state', 'UNAVAILABLE') != 'UNAVAILABLE'
            for entry in client.get_model_repository_index()):
    if time.monotonic() >= deadline:
      raise TimeoutError(f'Triton model {model} did not unload within {unload_timeout} seconds')
    time.sleep(0.1)
  try: shutil.rmtree(TRITON_MODEL_REPOSITORY / model)
  except FileNotFoundError: pass
  for f in Path("/dev/shm").glob(f"{model}_*.parameters"):
    f.unlink(missing_ok=True)

def unload_triton_models(client: InferenceServerClient, model: Optional[str] = None):
  for model_stats in get_triton_inference_stats(client):
    if model is None or model == model_stats['name']:
      print(f"Unloading {model_stats['name']}")
      unload_triton_model(client, model_stats['name'])

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
  for model in get_triton_inference_stats(triton_client):
    last_inference_time = model['last_inference']//1000
    try: model_mtime = Path(TRITON_MODEL_REPOSITORY / model['name'] / '1').stat().st_mtime
    except FileNotFoundError: model_mtime = 0
    try: parameters = triton_client.get_model_config(model['name']).get('parameters', {})
    except InferenceServerException: continue
    model_stale_after_seconds = float(
      parameters.get(TRITON_MODEL_STALE_AFTER_SECONDS_PARAMETER, {}).get('string_value', 30*60))
    if model['name'] != keep_model_name and (
      time.time() - max(last_inference_time, model_mtime) > model_stale_after_seconds):
      with redis_client.lock(model['name'], timeout=10*60):
        unload_triton_model(triton_client, model['name'])

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
