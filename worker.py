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
import traceback
import subprocess
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
from miniray.lib.helpers import Limits, desc, GB_TO_BYTES, TASK_TIMEOUT_GRACE_SECONDS, JOB_CACHE_SIZE
from miniray.lib.uv import sync_venv_cache, cleanup_venvs
from miniray import MinirayResultHeader, MinirayTask, JobMetadata, get_metadata_key

class Task:
  def __init__(self, task: MinirayTask, limits: Limits, proc_index: int,
               resource_manager: ResourceManager, r_master: redis.StrictRedis, r_results: redis.StrictRedis,
               job_metadata: JobMetadata, venv_cache: LRU, triton_client):
    self.task = task
    self.limits = limits
    self.proc_index = proc_index
    self.rm = resource_manager
    self.r_master = r_master
    self.r_results = r_results
    self.job_metadata = job_metadata
    self.venv_cache = venv_cache
    self.triton_client = triton_client

    # Process state
    self.proc = None
    self.alloc_id = None
    self.task_gid = None
    self.cgroup_name = None
    self.result_file = None
    self.start_time = None
    self.tmp_dir = None
    self.venv_dir = None

    # Result state (populated by check_done)
    self._done = False
    self._timed_out = False
    self._task_result = b''
    self._error = None

  @property
  def job(self):
    return self.task.job

  @property
  def task_uuid(self):
    return self.task.uuid

  def init(self) -> bool:
    try:
      if not self.job_metadata.valid:
        self._done = True
        self._error = ("InvalidJobError", "No valid JobMetadata, key was probably missing")
        return False

      try:
        ensure_venv(self.job, self.job_metadata.codedir, self.venv_cache)
        self.venv_dir = self.venv_cache[self.job]
      except (ValueError, AssertionError) as e:
        self._done = True
        self._error = ("VenvError", f"{type(e).__name__}:{e}")
        return False

      # Set start time tracking in Redis
      self.r_master.set(f'{self.task_uuid}-start',
                        json.dumps([self.job, WORKER_ID, time.time() + self.limits.timeout_seconds + TASK_TIMEOUT_GRACE_SECONDS]),
                        ex=7*24*3600)

      # Fetch function if needed
      if self.task.function_ptr:
        self.pickled_fn = self.r_master.get(self.task.function_ptr)
        if self.pickled_fn is None:
          self._done = True
          self._error = ("CacheMissError", f"Cached function {self.task.function_ptr} not found in redis")
          return False
      else:
        self.pickled_fn = base64.b64decode(self.task.pickled_fn)
      self.pickled_args = base64.b64decode(self.task.pickled_args)

      self.alloc_id = f"proc{self.proc_index:0>3}"
      self.task_gid = grp.getgrnam(self.alloc_id).gr_gid if not DOCKER_CONTAINER else TASK_UID
      self.cgroup_name = os.path.join(CGROUP_NODE, self.alloc_id) if not DOCKER_CONTAINER else ""
      mem_limit_bytes = int((self.limits.memory or 1) * GB_TO_BYTES)
      self.tmp_dir = get_tmp_dir_for_task(self.alloc_id)

      # Get allocation info from resource manager
      self.numa_node = self.rm.get_numa_node(self.task_uuid)
      self.big_gpu_id, self.small_gpu_id = self.rm.get_gpu_ids(self.task_uuid)

      if not DOCKER_CONTAINER:
        cgroup_create(self.cgroup_name)
        cgroup_set_numa_nodes(self.cgroup_name, [self.numa_node])
        cgroup_set_memory_limit(self.cgroup_name, mem_limit_bytes)
      create_tmp_dir(self.tmp_dir, TASK_UID, self.task_gid)

      self.result_file = os.path.join(self.tmp_dir, "task_result")
      return True
    except BaseException as e:
      traceback.print_exc()
      self._done = True
      self._error = (type(e).__name__, traceback.format_exc())
      return False

  def start(self) -> bool:
    try:
      task_extra_groups = ["video"] + ["docker"] if not DOCKER_CONTAINER else []

      # CPU tasks only get a GPU if they reserve GPU memory
      cuda_visible_devices = []
      if self.big_gpu_id is not None:
        cuda_visible_devices.append(str(self.big_gpu_id))
      if self.small_gpu_id is not None:
        cuda_visible_devices.append(str(self.small_gpu_id))

      p_env = {
        **os.environ,
        'NO_PROGRESS': '1',
        'CUPY_CACHE_DIR': CUPY_CACHE_DIR,
        'CUDA_VISIBLE_DEVICES': ','.join(cuda_visible_devices),
        'USER': 'batman',
        'HOME': '/home/batman',
        'TASK_UID': str(TASK_UID),
        'TASK_CGROUP': self.cgroup_name,
        'TMPDIR': self.tmp_dir,
        'CACHE_ROOT': os.path.join(self.tmp_dir, "index_cache"),
        'PARAMS_ROOT': os.path.join(self.tmp_dir, "params"),
        'LOG_ROOT': os.path.join(self.tmp_dir, "media/0/realdata"),
        'GNSS_CACHE_DIR': os.path.join(self.tmp_dir, "gnss_cache"),
        'CDDIS_BASE_URL': "http://gnss-cache.comma.internal:8082/gnss-data",
        'CDDIS_HOURLY_BASE_URL': "http://gnss-cache.comma.internal:8082/gnss-data-hourly",
        'ENABLE_MODEL_CACHE': str(int(not TRITON_SERVER_ENABLED)),
        'RESULT_FILE': self.result_file,
      }
      python3_exe = os.path.join(self.venv_dir, "bin/python3")

      cgroup_controllers = ",".join(CGROUP_CONTROLLERS)
      p_args = ["cgexec", "-g", f"{cgroup_controllers}:/{self.cgroup_name}", "--sticky", python3_exe, os.path.join(SCRIPT_DIR, "lib/worker_task.py")]
      if DEBUG: print("[worker]", " ".join(p_args))

      self.proc = subprocess.Popen(p_args, user=0, group=self.task_gid, extra_groups=task_extra_groups, cwd=EMPTY_DIR, env=p_env,
                                   start_new_session=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      assert self.proc.stdin is not None
      self.proc.stdin.write(self.pickled_fn)
      self.proc.stdin.write(self.pickled_args)
      self.proc.stdin.close()
      self.proc.stdin = None
      self.start_time = time.time()
      return True
    except BaseException as e:
      traceback.print_exc()
      self._done = True
      self._error = (type(e).__name__, traceback.format_exc())
      return False

  def check_done(self, exiting=False) -> bool:
    if self._done:
      return True

    self._timed_out = time.time() > self.start_time + self.limits.timeout_seconds

    # Wait for the process to terminate
    if self.proc.returncode is None:
      pid, returncode = os.waitpid(self.proc.pid, os.WNOHANG)
      if not pid and not self._timed_out:
        return False  # still waiting
      self.proc.returncode = returncode

    # Kill the process group and wait for it to terminate
    if self.proc.returncode is not None or self._timed_out:
      if not reap_process(self.proc):
        return False  # still waiting

    # Collect stdout/stderr
    stdout, stderr = self.proc.communicate()
    if stdout:
      print(stdout.decode())
    if stderr:
      print(stderr.decode())

    # Read result file
    try:
      with open(self.result_file, 'rb') as f:
        self._task_result = f.read()
    except FileNotFoundError:
      self._task_result = b''

    # Determine result/error state
    if self._timed_out:
      self._error = ("TimeoutError", f"TimeoutError: task timed out after {self.limits.timeout_seconds} seconds")
    elif self.proc.returncode != 0 and not exiting:
      error_type = f"ChildProcessError<{self.proc.returncode}>"
      self._error = (error_type, f"{error_type}: task died with result code {self.proc.returncode}")
    elif self.proc.returncode == 0 and len(self._task_result) > 0:
      self._error = None
    elif self.proc.returncode == 0:
      self._error = ("NoResultError", "Task completed but produced no result")
    else:
      self._error = None  # exiting case

    self._done = True
    return True

  def finish(self, ignore_errors=False):
    """Push results to Redis and cleanup. This is the ONLY place that pushes to Redis."""
    # Collect stats (must happen before cgroup cleanup)
    if self.start_time is not None:
      task_gpu_stats = get_gpu_stats(self.proc.pid, [gpu.handle for gpu in self.rm.gpus])
      task_run_time = time.time() - self.start_time
      task_cpu_time = get_cgroup_cpu_usage(self.cgroup_name)
      task_gpu_time = get_gpu_utilization(task_gpu_stats) * task_run_time
      task_memory_gb = get_cgroup_mem_usage(self.cgroup_name) * 1e-9
      task_gpu_memory_gb = get_gpu_mem_usage(task_gpu_stats) * 1e-9
      statsd.event("pipeline.worker.task_done", runtime=task_run_time, cpu=task_cpu_time, gpu=task_gpu_time, memory=task_memory_gb, gpu_memory=task_gpu_memory_gb, tags={'task_id': self.job})
      print(f"[worker] finished miniray task from job {self.job} stats: elapsed={task_run_time:0.2f}s cpu={task_cpu_time:0.2f}s gpu={task_gpu_time:0.2f}s mem={task_memory_gb:0.2f}GB gpumem={task_gpu_memory_gb:0.2f}GB")

    if self._error:
      error_type, error_msg = self._error
      result_header = MinirayResultHeader(self.job, False, HOST_NAME, error_type, error_msg, self.task_uuid)
      self.r_results.lpush(f'fq-{self.job}', json.dumps(result_header))
      self.r_master.delete(f'{self.task_uuid}-start')
      statsd.event('pipeline.worker.task_error', tags={'task_id': self.job, 'type': error_type})
    elif len(self._task_result) > 0:
      success_marker = self._task_result[0:1]
      payload = self._task_result[1:]

      if success_marker == b'\x00':
        result_header = MinirayResultHeader(self.job, True, HOST_NAME, "", "", self.task_uuid)
        self.r_results.lpush(f'fq-{self.job}', json.dumps(result_header).encode() + b'\x00' + payload)
      else:
        error_type, error_desc = json.loads(payload)
        statsd.event('pipeline.worker.task_error', tags={'task_id': self.job, 'type': error_type})
        result_header = MinirayResultHeader(self.job, False, HOST_NAME, error_type, error_desc, self.task_uuid)
        self.r_results.lpush(f'fq-{self.job}', json.dumps(result_header))

      self.r_results.expire(f'fq-{self.job}', 86400)  # extend availability for 24 hours
      self.r_master.delete(f'{self.task_uuid}-start')

    # Cleanup cgroups, shared memory, and temp directories
    if self.alloc_id is not None:
      while True:
        try:
          if not DOCKER_CONTAINER:
            cgroup_kill(self.cgroup_name, recursive=True)
            cgroup_delete(self.cgroup_name, recursive=True)
          break
        except Exception as e:
          print(f"[worker] {self.cgroup_name} cgroup cleanup failed: {desc(e)}")
          if ignore_errors:
            break
          time.sleep(1)

      while True:
        try:
          cleanup_shm_by_gid(self.alloc_id, self.triton_client, self.task_gid)
          break
        except Exception as e:
          print(f"[worker] {self.cgroup_name} /dev/shm cleanup failed: {desc(e)}")
          if ignore_errors:
            break
          time.sleep(1)

    self.rm.release(self.task_uuid)

DOCKER_CONTAINER = os.path.exists("/.dockerenv")
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
MINIRAY_TARGET_NAME = "<remote-function>"

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
             r_results: redis.StrictRedis, job: str, job_metadatas: dict[str, JobMetadata], venvs: LRU,
             proc_index: int, triton_client) -> Optional[Task]:
  limits = Limits(**job_metadatas[job].limits)
  temp_key = f"{job}-pending"
  try:
    resource_manager.consume(limits, job, task_uuid=temp_key)
  except ResourceLimitError as e:
    print(f"[worker] {MINIRAY_TARGET_NAME} resource limit: {desc(e)}")
    return None

  raw_task = r_tasks.rpop(job)
  if not raw_task:  # something else grabbed the last task
    resource_manager.release(temp_key)
    return None

  miniray_task = MinirayTask(*json.loads(raw_task))
  resource_manager.rekey(temp_key, miniray_task.uuid)

  return Task(miniray_task, limits, proc_index, resource_manager, r_master, r_results, job_metadatas[job], venvs, triton_client)


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
  rm = ResourceManager(triton_client=triton_client)

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

  sigterm_handler = SigTermHandler(callback=sig_callback)
  backoff = ExponentialBackoff(SLEEP_TIME_MAX, DEBUG)

  r_master = redis.StrictRedis(host=REDIS_HOST, port=6379, db=1)
  r_tasks = redis.StrictRedis(host=REDIS_HOST, port=6379, db=4)
  r_results = redis.StrictRedis(host=REDIS_HOST, port=6379, db=5)

  os.nice(1)

  procs: dict[int, Optional[Task]] = dict.fromkeys(range(sum(rm.cpu_totals.values())))

  while not sigterm_handler.raised:
    r_master.set(ACTIVE_KEY, 1, ex=SLEEP_TIME_MAX+1)
    backoff.sleep()

    jobs = sorted(key.decode() for key in r_tasks.keys(f"*{PIPELINE_QUEUE}"))
    update_job_metadatas(r_master, jobs, job_metadatas)
    current_gpu_job = get_globally_scheduled_job(r_master, jobs, job_metadatas)
    for i in procs.keys():
      # check if task is done and handle completion
      if procs[i] and procs[i].check_done():
        procs[i].finish()
        procs[i] = None

      # if still working skip
      if procs[i] is not None:
        continue

      # schedule new task if slot is free
      task = None
      if current_gpu_job is not None:
        task = get_task(rm, r_master, r_tasks, r_results, current_gpu_job, job_metadatas, venvs, i, triton_client)
      if task is None:
        job = get_randomly_scheduled_job(r_master, jobs, job_metadatas)
        if job is not None:
          task = get_task(rm, r_master, r_tasks, r_results, job, job_metadatas, venvs, i, triton_client)
      if task is None:
        continue

      print(f"[worker] starting miniray task from job {task.job} on proc{i}")
      if task.init() and task.start():
        procs[i] = task
        backoff.reset()
      else:
        task.finish()

  # send sigterm to all remaining processes
  for i in procs.keys():
    if procs[i] and procs[i].proc:
      os.killpg(procs[i].proc.pid, signal.SIGTERM)

  # wait for tasks to finish
  while any(procs.values()):
    for i in procs.keys():
      if procs[i] and procs[i].check_done(exiting=True):
        procs[i].finish(ignore_errors=True)
        procs[i] = None
    time.sleep(1)


if __name__ == '__main__':
  main()
