import os
import sys
import glob
import json
import time
import shutil
import subprocess
from functools import wraps
from pathlib import Path
from typing import Optional, Any
from redis import StrictRedis
from miniray.lib.helpers import desc
from tenacity import retry, stop_after_attempt, wait_random
from tritonclient.http import InferenceServerClient

TRITON_REDIS_HOST = os.getenv('TRITON_REDIS_HOST', '127.0.0.1')
TRITON_SERVER_ADDRESS = os.getenv('TRITON_SERVER_ADDRESS', '127.0.0.1:8000')
TRITON_MODEL_REPOSITORY = Path(os.getenv('TRITON_MODEL_REPOSITORY', '/dev/shm/model-repository'))

NOT_READY_MSG = "Triton server is not yet ready! If this persists for more than a few seconds, try restarting the triton server"
CONNECTION_ERR_MSG = f"""
Unable to connect to the triton server at {TRITON_SERVER_ADDRESS}.
 - If this occurs on your workstation, make sure the triton server is active.
 - If this occurs on a worker, it may indicate that the server has crashed. This can occur, for instance, if the 3080 TI falls off the bus.
""".strip()

def create_triton_client(url: str = TRITON_SERVER_ADDRESS, verbose: bool = False) -> InferenceServerClient:
  client = InferenceServerClient(url=url, verbose=verbose)
  try:
    assert client.is_server_live(), NOT_READY_MSG
  except ConnectionRefusedError:
    raise ConnectionRefusedError(CONNECTION_ERR_MSG)
  return client

@retry(stop=stop_after_attempt(3), wait=wait_random(1, 2), reraise=True)
def get_triton_inference_stats(client: InferenceServerClient):
  return client.get_inference_statistics()['model_stats']

@retry(stop=stop_after_attempt(3), wait=wait_random(1, 2), reraise=True)
def load_triton_model(client: InferenceServerClient, model: str, config: dict[str, Any]):
  return client.load_model(model, config=json.dumps(config))

def setup_triton_model(func):
  @wraps(func)
  def wrapper(*self, client: InferenceServerClient, model: str, redis: Optional[StrictRedis] = None) -> None:
      model_dir = TRITON_MODEL_REPOSITORY / model / '1'
      if client.is_model_ready(model):  # if the model is already loaded, bump the mtime and return
        mtime = time.time()
        try: os.utime(model_dir, (mtime, mtime))
        except OSError: pass
        return
      if redis is None:
        redis = StrictRedis(host=TRITON_REDIS_HOST, port=6379, db=8)
      with redis.lock(model, timeout=4*60):
        if client.is_model_ready(model):
          return  # check if the model is ready both before and after acquiring the lock
        shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        config = func(*self, model_dir)
        load_triton_model(client, model, config)
        assert client.is_model_ready(model)
  return wrapper

def unload_triton_model(client: InferenceServerClient, model: str):
  client.unload_model(model)
  try: shutil.rmtree(TRITON_MODEL_REPOSITORY / model)
  except FileNotFoundError: pass
  for f in glob.glob(f"/dev/shm/{model}_*.parameters"):
    Path(f).unlink(missing_ok=True)

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
def kill_triton_backend_stubs(gpu_bus_ids: Optional[list[str]] = None):
  nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name", "--format=csv,noheader"]).decode('utf-8')
  for line in nvidia_smi_output.strip().split('\n'):
    gpu_bus_id, pid, process_name = line.split(', ')
    if (not gpu_bus_ids or gpu_bus_id in gpu_bus_ids) and 'triton_python_backend_stub' in process_name:
      try: os.kill(int(pid), 15)
      except ProcessLookupError: pass

def get_triton_container_id() -> str:
  container_ids = subprocess.check_output(["docker", "ps", "--format", "{{.ID}}", "--filter", "name=tritonserver"]).decode('utf-8').strip()
  if not container_ids:
    raise RuntimeError("No tritonserver container found")
  return container_ids.split('\n')[0]

def cleanup_triton(triton_client, gpu_bus_ids):
  try:
    unload_triton_models(triton_client)
  except ConnectionRefusedError as e:
    print(f"[worker] could not connect to triton server: {desc(e)}")
  except Exception as e:
    print(f"[worker] error unloading triton models: {desc(e)}")
  try:
    kill_triton_backend_stubs(gpu_bus_ids)
  except Exception as e:
    print(f"[worker] error killing triton backend stubs: {desc(e)}")


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
