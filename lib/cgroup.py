import os
import time
from itertools import count

CGROUP_DELETE_RETRIES = 5

def _get_cgroup_path(name: str) -> str:
  return os.path.join("/sys/fs/cgroup/", name)


def _get_numa_cpu_list(numa_node):
  with open(f"/sys/devices/system/node/node{numa_node}/cpulist", "r") as f:
    return f.read().strip()


def _validate_permissions(cgroup_path: str):
  user_id = os.getuid()
  if user_id != 0 and os.stat(cgroup_path).st_uid != user_id:
    raise Exception(f"VALIDATION FAILED: cgroup {cgroup_path} not owned by {user_id}")


def cgroup_delete(name: str, recursive: bool=False) -> None:
  cgroup_path = _get_cgroup_path(name)
  if os.path.isdir(cgroup_path):
    if recursive:
      with os.scandir(cgroup_path) as it:
        for de in it:
          if de.is_dir():
            cgroup_delete(os.path.join(name, de.name), recursive)
    os.rmdir(cgroup_path)


def cgroup_create(name: str) -> None:
  cgroup_path = _get_cgroup_path(name)
  if os.path.isdir(cgroup_path):
    _validate_permissions(cgroup_path)
    return
  try:
    os.mkdir(cgroup_path, mode=0o755)
  except PermissionError as e:
    raise PermissionError(f"could not create cgroup, manually create with:\nsudo mkdir -p {cgroup_path} && sudo chown $USER:$USER {cgroup_path}") from e


def cgroup_set_numa_nodes(name: str, numa_nodes: list[int]) -> None:
  cgroup_path = _get_cgroup_path(name)
  cpu_lists = [_get_numa_cpu_list(node) for node in numa_nodes]
  with open(os.path.join(cgroup_path, "cpuset.cpus"), "w") as f:
    f.write(",".join(cpu_lists))
  with open(os.path.join(cgroup_path, "cpuset.mems"), "w") as f:
    f.write(",".join(map(str, numa_nodes)))


def cgroup_set_subcontrollers(name: str, controllers: list[str]) -> None:
  cgroup_path = _get_cgroup_path(name)
  with open(os.path.join(cgroup_path, "cgroup.subtree_control"), "w") as f:
    f.write(" ".join(f"+{c}" for c in controllers))


def cgroup_set_memory_limit(name: str, limit_in_bytes: int) -> None:
  cgroup_path = _get_cgroup_path(name)
  with open(os.path.join(cgroup_path, "memory.max"), "w") as f:
    f.write(str(limit_in_bytes or "max"))
  with open(os.path.join(cgroup_path, "memory.swap.max"), "w") as f:
    f.write("0")


def cgroup_add_pid(name: str, pid: int) -> None:
  cgroup_path = _get_cgroup_path(name)
  with open(os.path.join(cgroup_path, "cgroup.procs"), "w") as f:
    f.write(str(pid))


def cgroup_kill(name: str) -> None:
  cgroup_path = _get_cgroup_path(name)
  with open(os.path.join(cgroup_path, "cgroup.kill"), "w") as f:
    f.write("1")


def cgroup_clear_all_children(name: str) -> None:
  cgroup_path = _get_cgroup_path(name)
  cgroup_kill(name)
  with os.scandir(cgroup_path) as it:
    for de in it:
      if de.is_dir():
        child_cgroup = os.path.join(name, de.name)
        for attempt in count():
          try:
            cgroup_delete(child_cgroup, recursive=True)
            break
          except OSError as e:
            if e.errno != 16 or attempt >= CGROUP_DELETE_RETRIES:  # errno 16 = device or resource busy, try again
              raise
            time.sleep(1)
