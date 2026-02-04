#!/usr/bin/env python
import os

# prevent lots of threads from being started when importing task classes
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import random
import json
import time
import base64
import redis
import signal
import socket
import hashlib
import platform
import traceback
import subprocess
import collections
import grp
import stat
import shutil
import numpy as np
from typing import Optional
from tritonclient.http import InferenceServerClient
from lru import LRU

from miniray.lib.cgroup import cgroup_create, cgroup_set_subcontrollers, cgroup_set_memory_limit, cgroup_set_numa_nodes, cgroup_kill, cgroup_delete, cgroup_clear_all_children
from miniray.lib.sig_term_handler import SigTermHandler
from miniray.lib.resource_manager import ResourceManager, ResourceLimitError
from miniray.lib.worker_helpers import ExponentialBackoff
from miniray.lib.triton_helpers import TRITON_SERVER_ADDRESS
from miniray.lib.system_helpers import get_cgroup_cpu_usage, get_cgroup_mem_usage, get_gpu_stats, get_gpu_mem_usage, get_gpu_utilization
from miniray.lib.statsd_helpers import statsd
from miniray.lib.helpers import Limits, desc, GB_TO_BYTES, TASK_TIMEOUT_GRACE_SECONDS, MEMORY_LIMIT_HEADROOM, JOB_CACHE_SIZE, JOB_BLOCK_SECONDS
from miniray.lib.uv import sync_venv_cache, cleanup_venvs
from miniray import MinirayResultHeader, MinirayTask, JobMetadata, get_metadata_key

ProcTask = collections.namedtuple("ProcTask", ["proc", "job", "alloc_id", 'task_gid', "cgroup_name", "limits",
                                               "task_uuid", "start_time", "result_file"])

DOCKER_CONTAINER = os.path.exists("/.dockerenv")

# note that /sys/devices/virtual/dmi/id/sys_vendor is empty on some of our dell servers
HOST = platform.node()
HOST_NAME = socket.gethostname()
TASK_UID = int(os.getenv("TASK_UID", "1000")) if not DOCKER_CONTAINER else os.getuid()
DEBUG = os.getenv("DEBUG_WORKER", None)
REDIS_HOST = os.getenv('REDIS_HOST', 'redis.comma.internal')
PIPELINE_QUEUE = os.getenv('PIPELINE_QUEUE', 'local'+"-"+HOST_NAME)   # override this in systemd
SLEEP_TIME_MAX = int(os.getenv('SLEEP_TIME_MAX', '2'))

EMPTY_DIR = "/tmp/empty" # Run worker_task.py from an empty directory so relative path lookups don't hit code.nfs
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CGROUP_NODE = "worker"
CGROUP_CONTROLLERS = ["cpu", "cpuset", "memory"]
WORKER_ID = HOST_NAME
ACTIVE_KEY = f"active:{PIPELINE_QUEUE}:{WORKER_ID}"
SUSPEND_KEY = f"suspend:{WORKER_ID}"
MINIRAY_TARGET_NAME = "<remote-function>"
MEM_LIMIT = 0.85

BLOCK_JOB_KEY_PREFIX = "block:"

TMP_DIR_ROOT = os.path.join("/dev/shm/tmp" if not DOCKER_CONTAINER else "/tmp", CGROUP_NODE)
# you need a really good reason to use a global directory shared across all tasks
# (normally you should use the tmp directory that is cleaned up after every task)
CUPY_CACHE_DIR = os.path.join(TMP_DIR_ROOT, "cupy")
TRITON_SERVER_ENABLED = int(os.getenv('TRITON_SERVER_ENABLED', '0'))


def setup_global_dirs():
  os.makedirs(EMPTY_DIR, exist_ok=True)
  os.chmod(EMPTY_DIR, 0o555)

  if os.path.exists(TMP_DIR_ROOT):
    shutil.rmtree(TMP_DIR_ROOT)
  os.makedirs(TMP_DIR_ROOT)

  os.makedirs(CUPY_CACHE_DIR)
  os.chown(CUPY_CACHE_DIR, TASK_UID, TASK_UID)
  os.chmod(CUPY_CACHE_DIR, 0o755)

def create_tmp_dir(path, uid, gid):
  if os.path.exists(path):
    shutil.rmtree(path)
  os.makedirs(path)
  os.chown(path, uid, gid)

def get_tmp_dir_for_task(alloc_id):
  return os.path.join(TMP_DIR_ROOT, alloc_id)

def cleanup_shm_by_gid(alloc_id, triton_client, gid):
  with os.scandir("/dev/shm") as it:
    shm_entries = [(de, de.stat(follow_symlinks=False)) for de in it]
    shm_entries_for_gid = [(de, s) for de, s in shm_entries if s.st_gid == gid]

  if TRITON_SERVER_ENABLED and len(shm_entries_for_gid) > 0:
    try:
      triton_shm_entries = {x['name'] for x in triton_client.get_system_shared_memory_status()}
      for de, s in shm_entries_for_gid:
        if de.name in triton_shm_entries and not stat.S_ISDIR(s.st_mode):
          triton_client.unregister_system_shared_memory(de.name)
    except ConnectionRefusedError as e:
      print(f"[worker] could not connect to triton server: {desc(e)}")

  for de, s in shm_entries_for_gid:
    if stat.S_ISDIR(s.st_mode):
      shutil.rmtree(de)
    else:
      os.unlink(de.path)

  tmp_dir = get_tmp_dir_for_task(alloc_id)
  if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)

def reap_process(proc):
  try:
    pid = -1
    while pid != 0:
      pid, _ = os.waitpid(-proc.pid, os.WNOHANG)
    if not hasattr(proc, 'sigterm_sent'):
      os.killpg(proc.pid, signal.SIGTERM)
      proc.sigterm_sent = time.time()
    elif proc.sigterm_sent + 30 > time.time():
      os.killpg(proc.pid, signal.SIGKILL)
    return False
  except (ChildProcessError, ProcessLookupError):
    return True  # all processes have exited

def push_error(r_master, r_result, job, task_uuid, host, error_type, error_msg):
  result_header = MinirayResultHeader(job, False, host, error_type, error_msg, task_uuid)
  r_result.lpush(f'fq-{job}', json.dumps(result_header))
  r_master.delete(f'{task_uuid}-start')
  statsd.event('pipeline.worker.task_error', tags={'task_id': job, 'type': error_type})

# Divide the interval [0, 1) amongst the available jobs, weighted by job priority.
# We want to make sure each task gets scheduled to at least one worker, so each job needs to have an interval
# of at least 1 / N, where N is the number of workers. Specifically, instead of normalizing by sum(weights),
# we want to normalize by x such that sum(max(1 / N, weights / x)) = 1.
# For example: if we have weights [1, 100, 1000] and N = 10, we find x = 1250
# to get weights = [0.1, 0.1, 0.8], which satisfies all(W >= 1 / N) and sum(W) == 1.
def get_job_intervals(raw_weights:list[int], N:int) -> list[float]:
  weights = all_weights = np.array(raw_weights[:N])  # we have N workers, so we can only schedule a maximum of N jobs at once
  x = 0
  while sum(weights) / N > x + 1e-7:  # small epsilon to fix rounding errors
    x = sum(weights) / N  # set x to its lower bound
    N -= sum(weights / x < 1)  # any weights less than 1 / N will be clamped, so remove these weights from both N and weights and keep looping
    weights = weights[weights / x >= 1]

  weights = np.maximum(1, all_weights / x)
  weights = np.cumsum(weights / weights.sum())
  return weights.tolist()  # type: ignore[no-any-return]

# To decide which job to accept, we do the following:
# - Find our position in the sorted list of N active workers, this will be a number in [0, N). Divide by N to get a point P in [0, 1).
# - Divide the interval [0, 1] amongst the available jobs, weighted by job priority
# - Find the job whose interval contains P, this will be the job we accept.
def get_globally_scheduled_job(r_master:redis.StrictRedis, jobs:list[str], job_metadatas: dict[str, JobMetadata]) -> Optional[str]:
  active_key = hashlib.md5(ACTIVE_KEY.encode()).hexdigest()  # we use the hash so machines with different compute capabilities are evenly distributed
  active_workers = sorted(hashlib.md5(k).hexdigest() for k in r_master.keys(f"active:{PIPELINE_QUEUE}:*"))
  if not jobs or active_key not in active_workers:
    return None

  P = active_workers.index(active_key) / len(active_workers)  # P is a point in [0, 1)
  job_weights = [job_metadatas[j].priority for j in jobs]
  job_intervals = get_job_intervals(job_weights, len(active_workers))
  job_index = next(i for i,end in enumerate(job_intervals) if end >= P + 1e-7)
  return jobs[job_index]

def get_randomly_scheduled_job(r_master:redis.StrictRedis, jobs:list[str], job_metadatas: dict[str, JobMetadata]) -> Optional[str]:
  # gpu jobs are only scheduled via the global scheduler
  jobs = [job for job in jobs if not Limits(**job_metadatas[job].limits).requires_gpu()]

  if not jobs:
    return None
  job_weights = [job_metadatas[j].priority for j in jobs]
  job = random.choices(jobs, weights=job_weights, k=1)[0]
  return job

def update_job_metadatas(r_master:redis.StrictRedis, jobs:list[str], job_metadatas:dict[str, JobMetadata]):
  for job in jobs:
    if job not in job_metadatas:
      raw_metadata = r_master.get(get_metadata_key(job))
      if raw_metadata is not None:
        job_metadatas[job] = JobMetadata(*json.loads(raw_metadata))
      else:
        job_metadatas[job] = JobMetadata(False, 1, "", "", Limits().asdict())

def get_task(resource_manager: ResourceManager, r_master: redis.StrictRedis, r_tasks: redis.StrictRedis,
             r_results: redis.StrictRedis, job: str, job_metadatas: dict[str, JobMetadata], venvs: LRU) -> Optional[MinirayTask]:
  if r_master.exists(BLOCK_JOB_KEY_PREFIX + job):
    return None

  # check resources before popping task
  limits = Limits(**job_metadatas[job].limits)
  temp_key = f"{job}-pending"
  try:
    resource_manager.consume(limits, job, task_uuid=temp_key)
  except ResourceLimitError as e:
    r_master.set(SUSPEND_KEY, desc(e), ex=SLEEP_TIME_MAX+1)
    print(f"[worker] {MINIRAY_TARGET_NAME} resource limit: {desc(e)}")
    return None

  raw_task = r_tasks.rpop(job)
  if not raw_task:
    resource_manager.release(temp_key)
    return None

  task = MinirayTask(*json.loads(raw_task))
  resource_manager.rekey(temp_key, task.uuid)

  if not job_metadatas[job].valid:
    push_error(r_master, r_results, task.job, task.uuid, HOST_NAME, "InvalidJobError", "No valid JobMetadata, key was probably missing")
    resource_manager.release(task.uuid)
    return None

  try:
    ensure_venv(job, job_metadatas[job].codedir, venvs)
  except (ValueError, AssertionError) as e:
    push_error(r_master, r_results, task.job, task.uuid, HOST_NAME, f"VenvError", f"{type(e).__name__}:{e}")
    resource_manager.release(task.uuid)
    return None

  r_master.set(f'{task.uuid}-start',
               json.dumps([task.job, WORKER_ID, time.time() + limits.timeout_seconds + TASK_TIMEOUT_GRACE_SECONDS]),
               ex=7*24*3600)
  return task

def start_worker_task(task: MinirayTask, limits: Limits, i, rm, r_master, r_results, venv_dir):
  job = task.job
  task_uuid = task.uuid

  if task.function_ptr:
    pickled_fn = r_master.get(task.function_ptr)
    #TODO this should raise a task error
    assert pickled_fn is not None, f"Cached function {task.function_ptr} not found in redis"
  else:
    pickled_fn = base64.b64decode(task.pickled_fn)
  pickled_args = base64.b64decode(task.pickled_args)

  try:
    alloc_id = f"proc{i:0>3}"
    task_gid = grp.getgrnam(alloc_id).gr_gid if not DOCKER_CONTAINER else TASK_UID
    task_extra_groups = ["video"] + ["docker"] if not DOCKER_CONTAINER else []
    cgroup_task = os.path.join(CGROUP_NODE, alloc_id) if not DOCKER_CONTAINER else ""
    mem_limit_bytes = int((limits.memory or 1) * MEMORY_LIMIT_HEADROOM * GB_TO_BYTES)
    tmp_dir = get_tmp_dir_for_task(alloc_id)

    # Get allocation info from resource limiter
    numa_node = rm.get_numa_node(task_uuid)
    big_gpu_id, small_gpu_id = rm.get_gpu_ids(task_uuid)

    if not DOCKER_CONTAINER: # cgroup fs mounted read-only inside docker
      cgroup_create(cgroup_task)
      cgroup_set_numa_nodes(cgroup_task, [numa_node])
      cgroup_set_memory_limit(cgroup_task, mem_limit_bytes)
    create_tmp_dir(tmp_dir, TASK_UID, task_gid)

    # cpu tasks only get a GPU if they reserve GPU memory
    cuda_visible_devices = []
    if big_gpu_id is not None:
      cuda_visible_devices.append(str(big_gpu_id))
    if small_gpu_id is not None:
      cuda_visible_devices.append(str(small_gpu_id))

    result_file = os.path.join(tmp_dir, "task_result")

    p_env = {
      **os.environ,
      'NO_PROGRESS': '1',
      'CUPY_CACHE_DIR': CUPY_CACHE_DIR,
      'CUDA_VISIBLE_DEVICES': ','.join(cuda_visible_devices),
      'USER': 'batman',
      'HOME': '/home/batman',
      'TASK_UID': str(TASK_UID),
      'TASK_UUID': task_uuid,
      'TASK_CGROUP': cgroup_task,
      'TMPDIR': tmp_dir,
      'CACHE_ROOT':  os.path.join(tmp_dir, "index_cache"),
      'PARAMS_ROOT': os.path.join(tmp_dir, "params"),
      'LOG_ROOT': os.path.join(tmp_dir, "media/0/realdata"),
      'GNSS_CACHE_DIR': os.path.join(tmp_dir, "gnss_cache"),
      'CDDIS_BASE_URL': "http://gnss-cache.comma.internal:8082/gnss-data",
      'CDDIS_HOURLY_BASE_URL': "http://gnss-cache.comma.internal:8082/gnss-data-hourly",
      'ENABLE_MODEL_CACHE': str(int(not TRITON_SERVER_ENABLED)),
      'RESULT_FILE': result_file,
    }
    python3_exe = os.path.join(venv_dir, "bin/python3")

    cgroup_controllers = ",".join(CGROUP_CONTROLLERS)
    p_args = ["cgexec", "-g", f"{cgroup_controllers}:/{cgroup_task}", "--sticky", python3_exe, os.path.join(SCRIPT_DIR, "lib/worker_task.py")]
    if DEBUG: print("[worker]", " ".join(p_args))
    # start a new process so that all modules are imported fresh
    # (ensuring all code changes to task classes and their deps get picked up)
    p = subprocess.Popen(p_args, user=0, group=task_gid, extra_groups=task_extra_groups, cwd=EMPTY_DIR, env=p_env,
                         start_new_session=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert p.stdin is not None
    p.stdin.write(pickled_fn)
    p.stdin.write(pickled_args)
    p.stdin.close()  # close stdin so child's read() gets EOF
    p.stdin = None
    return ProcTask(p, job, alloc_id, task_gid, cgroup_task, limits, task_uuid, time.time(), result_file)
  except BaseException as e:
    traceback.print_exc()
    push_error(r_master, r_results, job, task_uuid, HOST_NAME, type(e).__name__, traceback.format_exc())
    rm.release(task_uuid)

def check_task_completion(pt, r_master, r_results, rm, exiting=False) -> bool:
  timed_out = time.time() > pt.start_time + pt.limits.timeout_seconds

  # wait for the process to terminate
  if pt.proc.returncode is None:
    pid, returncode = os.waitpid(pt.proc.pid, os.WNOHANG)
    if not pid and not timed_out:
      return False  # still waiting
    pt.proc.returncode = returncode

  # kill the process group and wait for it to terminate
  if pt.proc.returncode is not None or timed_out:
    if not reap_process(pt.proc):
      return False  # still waiting

  #TODO This needs to be sent back
  stdout, stderr = pt.proc.communicate()
  if stdout:
    print(stdout.decode())
  if stderr:
    print(stderr.decode())

  try:
    with open(pt.result_file, 'rb') as f:
      task_result = f.read()
  except FileNotFoundError:
    task_result = b''

  task_gpu_stats = get_gpu_stats(pt.proc.pid, [gpu.handle for gpu in rm.gpus])
  task_run_time = time.time() - pt.start_time
  task_cpu_time = get_cgroup_cpu_usage(pt.cgroup_name)
  task_gpu_time = get_gpu_utilization(task_gpu_stats) * task_run_time
  task_memory_gb = get_cgroup_mem_usage(pt.cgroup_name) * 1e-9
  task_gpu_memory_gb = get_gpu_mem_usage(task_gpu_stats) * 1e-9

  statsd.event("pipeline.worker.task_done", runtime=task_run_time, cpu=task_cpu_time, gpu=task_gpu_time, memory=task_memory_gb, gpu_memory=task_gpu_memory_gb, tags={'task_id': pt.job})
  print(f"[worker] finished miniray task from job {pt.job} stats: elapsed={task_run_time:0.2f}s cpu={task_cpu_time:0.2f}s gpu={task_gpu_time:0.2f}s mem={task_memory_gb:0.2f}GB gpumem={task_gpu_memory_gb:0.2f}GB")

  # returncode 9 means something really bad happened, usually an OOM exception, so block the job for a while
  if pt.proc.returncode in (9, -9):
    r_master.set(BLOCK_JOB_KEY_PREFIX + pt.job, 1, ex=JOB_BLOCK_SECONDS)

  if timed_out:
    push_error(r_master, r_results, pt.job, pt.task_uuid, HOST_NAME, "TimeoutError", f"TimeoutError: task timed out after {pt.limits.timeout_seconds} seconds")
  elif pt.proc.returncode != 0 and not exiting:
    error_type = f"ChildProcessError<{pt.proc.returncode}>"
    push_error(r_master, r_results, pt.job, pt.task_uuid, HOST_NAME, error_type, f"{error_type}: task died with result code {pt.proc.returncode}")
  elif pt.proc.returncode == 0 and len(task_result) > 0:
    success_marker = task_result[0:1]
    payload = task_result[1:]

    if success_marker == b'\x00':
      result_header = MinirayResultHeader(pt.job, True, HOST_NAME, "", "", pt.task_uuid)
      r_results.lpush(f'fq-{pt.job}', json.dumps(result_header).encode() + b'\x00' + payload)
    else:
      error_type, error_desc = json.loads(payload)
      statsd.event('pipeline.worker.task_error', tags={'task_id': pt.job, 'type': error_type})
      result_header = MinirayResultHeader(pt.job, False, HOST_NAME, error_type, error_desc, pt.task_uuid)
      r_results.lpush(f'fq-{pt.job}', json.dumps(result_header))

    r_results.expire(f'fq-{pt.job}', 86400)  # extend availability for 24 hours
    r_master.delete(f'{pt.task_uuid}-start')
  elif pt.proc.returncode == 0:
    # Process exited cleanly but no result data - unexpected
    push_error(r_master, r_results, pt.job, pt.task_uuid, HOST_NAME, "NoResultError", "Task completed but produced no result")

  return True

def cleanup_task(cgroup_task, alloc_id, task_gid, r_master, triton_client, ignore_errors=False):
  while True:
    try:
      if not DOCKER_CONTAINER: # cgroup fs mounted read-only inside docker
        cgroup_kill(cgroup_task, recursive=True)
        cgroup_delete(cgroup_task, recursive=True)
      break
    except Exception as e:
      r_master.set(SUSPEND_KEY, desc(e), ex=SLEEP_TIME_MAX)
      print(f"[worker] {cgroup_task} cgroup cleanup failed: {desc(e)}")
      if ignore_errors:
        break
      time.sleep(1)

  while True:
    try:
      cleanup_shm_by_gid(alloc_id, triton_client, task_gid)
      break
    except Exception as e:
      r_master.set(SUSPEND_KEY, desc(e), ex=SLEEP_TIME_MAX)
      print(f"[worker] {cgroup_task} /dev/shm cleanup failed: {desc(e)}")
      if ignore_errors:
        break
      time.sleep(1)

def sig_callback(signal):
  print(f"[worker] cleaning up on signal: {signal} ...")

def ensure_venv(job: str, codedir: str, venv_cache: LRU):
  if job not in venv_cache and os.path.exists(codedir):
    venv_dir = str(sync_venv_cache(codedir, TASK_UID, job))
    venv_cache[job] = venv_dir
    cleanup_venvs(TASK_UID, keep_venvs=list(venv_cache.keys()))
  assert job in venv_cache, "Failed to find venv in cache"


def main():
  setup_global_dirs()

  # NOTE: This won't attempt to connect to triton until a request is made
  triton_client = InferenceServerClient(TRITON_SERVER_ADDRESS, verbose=False) if TRITON_SERVER_ENABLED else None
  rm = ResourceManager(mem_limit_multiplier=MEM_LIMIT, triton_client=triton_client)

  venvs: LRU[str, str] = LRU(JOB_CACHE_SIZE)
  job_metadatas: LRU[str, JobMetadata] = LRU(JOB_CACHE_SIZE)

  print(f"[worker] REDIS:                 {REDIS_HOST}")
  print(f"[worker] QUEUE:                 {PIPELINE_QUEUE}")
  print(f"[worker] CPU COUNT:             {sum(rm.cpu_totals.values())}")
  print(f"[worker] RAM:                   {sum(rm.mem_totals.values())/1e9:.2f} GB")
  print(f"[worker] BIG GPU IDs:           {', '.join(str(gpu.index) for gpu in rm.big_gpus)}")
  print(f"[worker] BIG GPU TOTAL RAM:     {sum(gpu.memory for gpu in rm.big_gpus)/1e9:.2f} GB")
  print(f"[worker] SMALL GPU ID:          {', '.join(str(gpu.index) for gpu in rm.small_gpus)}")
  print(f"[worker] SMALL GPU RAM:         {sum(gpu.memory for gpu in rm.small_gpus)/1e9:.2f} GB")
  print(f"[worker] TRITON_SERVER_ENABLED: {TRITON_SERVER_ENABLED}")

  if not DOCKER_CONTAINER: # cgroup fs mounted read-only inside docker
    cgroup_create(CGROUP_NODE)
    cgroup_set_subcontrollers(CGROUP_NODE, CGROUP_CONTROLLERS)
    cgroup_set_memory_limit(CGROUP_NODE, sum(rm.mem_totals.values()))
    cgroup_set_numa_nodes(CGROUP_NODE, rm.cpu_totals.keys())
    cgroup_clear_all_children(CGROUP_NODE)

  procs = dict.fromkeys(range(sum(rm.cpu_totals.values())))
  sigterm_handler = SigTermHandler(callback=sig_callback)
  backoff = ExponentialBackoff(SLEEP_TIME_MAX, DEBUG)

  r_master = redis.StrictRedis(host=REDIS_HOST, port=6379, db=1)
  r_tasks = redis.StrictRedis(host=REDIS_HOST, port=6379, db=4)
  r_results = redis.StrictRedis(host=REDIS_HOST, port=6379, db=5)

  os.nice(1)

  r_master.delete(SUSPEND_KEY)
  while not sigterm_handler.raised:
    r_master.set(ACTIVE_KEY, 1, ex=SLEEP_TIME_MAX+1)
    backoff.sleep()

    jobs = sorted(key.decode() for key in r_tasks.keys(f"*{PIPELINE_QUEUE}"))
    update_job_metadatas(r_master, jobs, job_metadatas)
    current_gpu_job = get_globally_scheduled_job(r_master, jobs, job_metadatas)
    for i in procs.keys():
      # cleanup completed tasks
      if procs[i] and check_task_completion(procs[i], r_master, r_results, rm):
        cleanup_task(procs[i].cgroup_name, procs[i].alloc_id, procs[i].task_gid, r_master, triton_client)
        rm.release(procs[i].task_uuid)
        procs[i] = None

      # if still working skip
      if procs[i] is not None:
        continue

      # schedule new task if slot is free
      task = None
      if current_gpu_job is not None:
        task = get_task(rm, r_master, r_tasks, r_results, current_gpu_job, job_metadatas, venvs)
      if task is None:
        job = get_randomly_scheduled_job(r_master, jobs, job_metadatas)
        if job is not None:
          task = get_task(rm, r_master, r_tasks, r_results, job, job_metadatas, venvs)
      if task is None:
        continue

      print(f"[worker] starting miniray task from job {task.job} on proc{i}")
      limits = Limits(**job_metadatas[task.job].limits)
      procs[i] = start_worker_task(task, limits, i, rm, r_master, r_results, venvs[task.job])
      backoff.reset()

  # send sigterm to all remaining processes
  for i in procs.keys():
    if procs[i]:
      os.killpg(procs[i].proc.pid, signal.SIGTERM)

  # wait for tasks to finish
  while any(procs.values()):
    for i in procs.keys():
      if procs[i] and check_task_completion(procs[i], r_master, r_results, rm, exiting=True):
        cleanup_task(procs[i].cgroup_name, procs[i].alloc_id, procs[i].task_gid, r_master, triton_client, ignore_errors=True)
        procs[i] = None
    time.sleep(1)


if __name__ == '__main__':
  main()
