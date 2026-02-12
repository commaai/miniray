import os
import sys
import json
import time
import uuid
import socket
import base64
import logging
import subprocess
import threading
import traceback
import cloudpickle
import multiprocessing as mp

from dataclasses import dataclass, asdict, field, replace
from datetime import datetime
from collections import Counter, defaultdict
from concurrent.futures import Future, Executor as BaseExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from itertools import islice, chain
from pathlib import Path
from queue import Queue
from redis import StrictRedis, ConnectionError as RedisConnectionError
from tqdm import tqdm
from typing import Any, Callable, Iterable, Iterator, NamedTuple, Optional, Sequence, cast

from miniray.lib.helpers import Limits, extract_error, StreamLogger

MAX_ARG_STRLEN = 131071  # max length for unix string arguments, see https://stackoverflow.com/a/29802900
REDIS_HOST = os.getenv('REDIS_HOST', 'redis.comma.internal')
FORCE_LOCAL = bool(int(os.getenv("MINIRAY_FORCE_LOCAL", "0")))
NUM_LOCAL_WORKERS = int(os.getenv("MINIRAY_LOCAL_NUM_WORKERS", "1"))
DEFAULT_RESULT_PAYLOAD_TIMEOUT_SECONDS = 20 * 60
USE_MAIN_RESULT_REDIS = bool(int(os.getenv("USE_MAIN_RESULT_REDIS", "0")))
CACHE_ROOT = Path("/code.nfs/branches/caches")
REMOTE_QUEUE = 'remote_v2'
DEFAULT_LOGGER = StreamLogger('miniray', level=logging.INFO)

MISSING_RESULT_PAYLOAD_ERROR = (
  f"Did not find payload on worker redis. Results may be piling up and reader has fallen more than {DEFAULT_RESULT_PAYLOAD_TIMEOUT_SECONDS/60:.1f}"
  " minutes behind. If your results are small, consider a larger chunksize. If your results are big, consider multiple miniray executors.")

#TODO xx should be referenced here
XX_BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
XX_BASEPATH = Path(XX_BASEDIR)


class MinirayError(Exception):
  def __init__(self, exception_type: str, exception_desc: str, job: str, worker: str):
    super().__init__(f"Task execution failed: {job} [{worker}]\n{exception_desc}")
    self.exception_type = exception_type
    self.exception_desc = exception_desc
    self.job = job
    self.worker = worker

class MinirayTask(NamedTuple):
  uuid: str
  job: str
  function_ptr: str
  pickled_fn: str
  pickled_args: str

class JobMetadata(NamedTuple):
  valid: bool
  priority: int
  codedir: str
  executor_host: str
  limits: dict[str, Any]

class MinirayResultHeader(NamedTuple):
  job: str
  succeeded: bool
  worker: str
  exception_type: str
  exception_desc: str
  task_uuid: str

class MiniraySubTaskResult(NamedTuple):
  succeeded: bool
  exception_type: str
  exception_desc: str
  result: Any

@dataclass
class JobConfig:
  priority: int = 1
  job_name: str = 'unnamed'
  queue_name: str = REMOTE_QUEUE
  redis_host: str = REDIS_HOST
  codedir: Optional[str] = None
  use_local_codedir: bool = False
  limits: Limits = field(default_factory=Limits)

  def asdict(self):
    return asdict(self)

def get_metadata_key(job: str) -> str:
  return f'job-metadata:{job}'

def sync_local_codedir(job_desc: str) -> str:
  cache_name = f"{job_desc}_{socket.gethostname()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex}"
  # TODO dont hardcode XX
  cache_dir = CACHE_ROOT / cache_name / "xx"
  cache_dir.mkdir(parents=True, exist_ok=True)
  excludes: list[str] = []
  if (base_exclude := XX_BASEPATH / "training/.training_cache_exclude").exists():
    excludes.append(f"--exclude-from={base_exclude}")
  if (local_exclude := XX_BASEPATH / "training/.training_cache_exclude.local").exists():
    excludes.append(f"--exclude-from={local_exclude}")
  dest = cache_dir.relative_to(Path("/code.nfs")).as_posix()
  subprocess.check_call(["rsync", "-a", "--max-delete=0", "--copy-dest=/xx", "--info=progress2", *excludes, f"{XX_BASEPATH}/", f"rsync://app01:1026/code_nfs/{dest}/"])
  return str(cache_dir)

def _execute_batch(fn, *batch, **kwargs):
  results = []
  for args in batch:
    try:
      results.append(MiniraySubTaskResult(True, "", "", fn(*args, **kwargs)))
    except BaseException as e:
      results.append(MiniraySubTaskResult(False, type(e).__name__, traceback.format_exc(), None))
  return _wrap_result_local_redis(results, timeout_seconds=DEFAULT_RESULT_PAYLOAD_TIMEOUT_SECONDS)

def _wrap_result_local_redis(data: Any, timeout_seconds: int) -> tuple[str, str]:
  key = f"miniray-{uuid.uuid4()}"
  redis_result_host = REDIS_HOST if USE_MAIN_RESULT_REDIS else socket.gethostname()
  r = StrictRedis(host=redis_result_host, db=10)
  pipe = r.pipeline()
  pipe.lpush(key, cloudpickle.dumps(data))
  pipe.expire(key, timeout_seconds)
  pipe.execute()
  return (redis_result_host, key)

def _local_worker_init():
  if (seed := os.getenv("MINIRAY_LOCAL_SEED")) is not None:
    from miniray.lib.helpers import set_random_seeds
    set_random_seeds(int(seed))

class LocalExecutor(ProcessPoolExecutor):
  def __init__(self):
    ctx = mp.get_context("spawn")
    # separate processes per task to avoid leaking states (simulating a behaviour from distributed run)
    super().__init__(max_workers=NUM_LOCAL_WORKERS, mp_context=ctx, max_tasks_per_child=1, initializer=_local_worker_init)

  def fmap(self, fn: Callable, *iterables: Iterable[Any], chunksize: int = 1) -> Iterator[Future]:
    for args in zip(*iterables, strict=True):
      yield self.submit(fn, *args)


class Executor(BaseExecutor):
  def __new__(cls, *args, **kwargs):
    if FORCE_LOCAL or kwargs.pop('force_local', False):
      return LocalExecutor()
    return super().__new__(cls)

  def __init__(self, config: Optional[JobConfig] = None, **kwargs) -> None:
    limits = kwargs.pop('limits', {})
    config = JobConfig() if config is None else config
    config = replace(config, **kwargs)
    if isinstance(limits, dict):
      limits = replace(config.limits, **limits)
    config = replace(config, limits=limits)

    assert config.job_name.replace('.','').replace('-', '').replace('_', '').isalnum(), f'Invalid job name: {config.job_name}'
    job_desc = f"{config.job_name}_{str(uuid.uuid4())[:8]}"

    if config.codedir is not None:
      assert os.path.exists(config.codedir), f"codedir {config.codedir} does not exist"
      assert config.use_local_codedir is False, "can't specify both codedir and use_local_codedir"
    elif config.use_local_codedir:
      config = replace(config, codedir=sync_local_codedir(job_desc))
    else:
      config = replace(config, codedir=str(XX_BASEPATH))

    self.config = config

    assert self.config.codedir is not None
    self.codedir = self.config.codedir
    self.submit_queue_id = f'{job_desc}-{self.config.queue_name}'
    self.result_queue_id = f'fq-{self.submit_queue_id}'

    self._futures: dict[str, list[Future]] = {}
    self._submit_redis_master = StrictRedis(host=self.config.redis_host, port=6379, db=1, socket_keepalive=True)
    self._submit_redis_tasks = StrictRedis(host=self.config.redis_host, port=6379, db=4, socket_keepalive=True)
    self._result_redis = StrictRedis(host=self.config.redis_host, port=6379, db=5, socket_keepalive=True)
    self._shutdown_lock = threading.Lock()
    self._shutdown_reader_thread = False
    self._reader_thread: Optional[threading.Thread] = None
    self._last_resubmit_check = 0.0
    self.expiry_check_timer: float = time.time()
    self.no_work_found_cnt = 0

    job_metadata = JobMetadata(True, self.config.priority, self.codedir, socket.gethostname(), self.config.limits.asdict())
    self._submit_redis_master.set(get_metadata_key(self.submit_queue_id), json.dumps(job_metadata), ex=7*24*60*60)


  def __enter__(self):
    self._shutdown_reader_thread = False
    self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
    self._reader_thread.start()
    return super().__enter__()

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.shutdown()
    return False

  # API methods

  def shutdown(self, wait: bool = True, cancel_futures: bool = False):
    with self._shutdown_lock:
      if cancel_futures:
        for futures in self._futures.values():
          for future in futures:
            future.cancel()

      self._shutdown_reader_thread = True
      if wait and self._reader_thread is not None:
          self._reader_thread.join()
      self._futures.clear()
      self._submit_redis_tasks.delete(self.submit_queue_id)
      self._submit_redis_master.delete(get_metadata_key(self.submit_queue_id))

  def submit(self, fn: Callable, /, *args, **kwargs) -> Future:
    assert not self._shutdown_reader_thread, "Cannot submit new tasks after shutdown has started"
    future: Future = Future()
    task_uuid = str(uuid.uuid4())
    pickled_fn = cloudpickle.dumps(partial(_execute_batch, fn))
    task = self._pack_task('', pickled_fn, [args], kwargs, task_uuid)
    self._futures[task_uuid] = [future]
    self._submit_task([task])
    return future

  def map(self, fn: Callable, *iterables: Iterable[Any], timeout: Optional[float] = None, chunksize: int = 1) -> Iterator[Any]:
    if timeout is not None:
      raise NotImplementedError("Timeout arg is not supported. Use `fmap` instead to get a timeout per task.")
    # submit all tasks first, then resolve the results lazily
    futures = list(self.fmap(fn, *iterables, chunksize=chunksize))
    return (future.result() for future in futures)

  def fmap(self, fn: Callable, *iterables: Iterable[Any], chunksize: int = 1) -> Iterator[Future]:
    assert not self._shutdown_reader_thread, "Cannot submit new tasks after shutdown has started"

    # Instead of sending the function along with every request, we cache it in redis and send the cache key in its place
    pickled_fn = cloudpickle.dumps(partial(_execute_batch, fn))
    function_ptr = f'pickledfunc-{uuid.uuid4()}'
    self._submit_redis_master.set(function_ptr, pickled_fn, ex=7*24*60*60)

    submitted_queue: Queue[Optional[Future]] = Queue()
    writer_thread = threading.Thread(target=self._writer_loop, args=(submitted_queue, function_ptr, list(iterables), chunksize), daemon=True)
    writer_thread.start()
    while future := submitted_queue.get():
      yield future

  def get_submit_queue_size(self) -> int:
    return cast(int, self._submit_redis_tasks.llen(self.submit_queue_id))

  # Worker threads

  def _writer_loop(self, submitted_queue: Queue[Optional[Future]], function_ptr: str, iterables: list[Iterable[Any]], chunksize: int) -> None:
    try:
      args_iterator = zip(*iterables, strict=True)
      assert chunksize >= 1
      while batch := list(islice(args_iterator, chunksize)):
        task_uuid = str(uuid.uuid4())
        futures: list[Future] = [Future() for _ in batch]
        self._futures[task_uuid] = futures
        for future in futures:
          submitted_queue.put(future)

        try:
          self._submit_task([self._pack_task(function_ptr, b'', batch, {}, task_uuid)])
        except Exception as e:
          for future in futures:
            future.set_exception(e)

      submitted_queue.put(None)  # Signal the end of the stream
    except Exception:
      print("[ERROR] Uncaught error in miniray writer thread", file=sys.stderr)
      print(traceback.format_exc(), file=sys.stderr)
      sys.exit(1)

  def _reader_loop(self) -> None:
    while not self._shutdown_reader_thread or not all(future.done() for future in chain.from_iterable(self._futures.values())):
      try:
        if time.time() - self.expiry_check_timer > 10:
          self._check_expired_tasks()
        results = cast(list[bytes], self._result_redis.lpop(self.result_queue_id, count=1000) or [])
        for result in results:
          self._unpack_result(result)
          self.expiry_check_timer = time.time()
        time.sleep(0.1)
      except RedisConnectionError:
        print("[ERROR] Redis connection error in miniray reader thread. Retrying in 10 seconds...", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        time.sleep(10)

  def _check_expired_tasks(self) -> None:
    # TODO this is super jank. Worker should garantuee work keys are present, so no race conditions
    now = time.time()
    self.expiry_check_timer = now
    any_working = False
    for task_uuid in list(self._futures.keys())[:1000]:  # avoid blocking for too long
      start_data = cast(Optional[bytes], self._submit_redis_master.get(f"{task_uuid}-start"))
      if start_data is not None:
        any_working = True
        job, worker, expiry_time = json.loads(start_data)
        if now > float(expiry_time):
          self._submit_redis_master.delete(f"{task_uuid}-start")
          for future in self._futures.pop(task_uuid):
            future.set_exception(MinirayError("TimeoutError", "TimeoutError: task did not complete before expiry time", job, worker))
    if not any_working and self.get_submit_queue_size() == 0:
      self.no_work_found_cnt += 1
      if self.no_work_found_cnt >= 3:
        for task_uuid in list(self._futures.keys()):
          for future in self._futures.pop(task_uuid):
            future.set_exception(MinirayError("RuntimeError", "task got lost", "", ""))

  def _pack_task(self, function_ptr: str, pickled_fn: bytes, args: Sequence[Any], kwargs: dict[str, Any], task_uuid: str) -> bytes:
    pickled_args = cloudpickle.dumps((args, kwargs))
    if len(pickled_fn) + len(pickled_args) > MAX_ARG_STRLEN:
      raise RuntimeError(f"Can't send target, size ({len(pickled_fn) + len(pickled_args)}) exceeds max allowed length ({MAX_ARG_STRLEN})")
    task = MinirayTask(
      task_uuid,
      self.submit_queue_id,
      function_ptr,
      base64.b64encode(pickled_fn).decode('ascii'),
      base64.b64encode(pickled_args).decode('ascii'),
    )
    return json.dumps(task, ensure_ascii=False).encode('utf-8')

  def _unpack_result(self, res: bytes) -> None:
    dat = res.split(b"\x00", 1)
    header = MinirayResultHeader(*json.loads(dat[0]))
    if header.task_uuid not in self._futures:
      print(f"[ERROR] finished unstarted task: {header.task_uuid} [{header.worker}]", file=sys.stderr)
      return
    self._submit_redis_master.delete(f"{header.task_uuid}-start")
    futures = self._futures.pop(header.task_uuid)

    if header.succeeded:
      hostname, key = cloudpickle.loads(dat[1])
      r = StrictRedis(host=hostname, db=10)
      result_payload = cast(Optional[bytes], r.lpop(key))
      if result_payload is None:
        for future in futures:
          future.set_exception(MinirayError("MinirayError", MISSING_RESULT_PAYLOAD_ERROR, header.job, header.worker))
      else:
        subtasks = cloudpickle.loads(result_payload)
        for future, subtask in zip(futures, subtasks, strict=True):
          if subtask.succeeded:
            future.set_result(subtask.result)
          else:
            future.set_exception(MinirayError(subtask.exception_type, subtask.exception_desc, header.job, header.worker))
    else:
      for future in futures:
        future.set_exception(MinirayError(header.exception_type, header.exception_desc, header.job, header.worker))

  def _submit_task(self, batch: list[bytes]) -> None:
    self._submit_redis_tasks.lpush(f'{self.submit_queue_id}', *batch)

def log(iterable: Iterable[Future], logger: Any = DEFAULT_LOGGER, desc: str = 'running miniray tasks', **kwargs: Any) -> list[Any]:
  results = []
  statuses: Counter[str] = Counter()
  statuses_hosts = defaultdict(list)
  iterable = list(iterable)
  for future in tqdm(as_completed(iterable), total=len(iterable), desc=desc, **kwargs):
    try:
      result = future.result()
      statuses["Succeeded"] += 1
      results.append(result)
    except MinirayError as e:
      error = extract_error(e.exception_type)
      statuses[error] += 1
      statuses_hosts[error].append(e.worker)
      if error != "ColumnStoreException":
        logger.error(f"FAILED TASK {e.job} [{e.worker}]\n{e.exception_desc}")

  logger.info("\n\n=== Miniray job summary ===")
  logger.info(f"Total segments: {sum(statuses.values())}")
  for status, cnt in statuses.most_common():
    status_hosts = Counter(statuses_hosts[status]).most_common()
    status_hosts_str = ' '.join(str(x) for x in status_hosts[:3]) + (' ...' if len(status_hosts) > 3 else '')
    logger.info(f"  {status} ({cnt})  {status_hosts_str}")

  return results
