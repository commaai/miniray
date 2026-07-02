"""Minimal ctypes wrapper for the Landlock LSM (Linux >= 5.13).

Used to sandbox task processes: filesystem writes are denied everywhere except an explicit
whitelist, while reads and execution stay unrestricted. The sandbox is inherited by all child
processes and cannot be lifted once applied.

No dependencies outside the standard library, so it can be imported from any job venv.
"""
from __future__ import annotations

import ctypes
import os
import stat
from collections.abc import Iterable

# constants from linux/landlock.h
LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1

ACCESS_FS_EXECUTE = 1 << 0
ACCESS_FS_WRITE_FILE = 1 << 1
ACCESS_FS_READ_FILE = 1 << 2
ACCESS_FS_READ_DIR = 1 << 3
ACCESS_FS_REMOVE_DIR = 1 << 4
ACCESS_FS_REMOVE_FILE = 1 << 5
ACCESS_FS_MAKE_CHAR = 1 << 6
ACCESS_FS_MAKE_DIR = 1 << 7
ACCESS_FS_MAKE_REG = 1 << 8
ACCESS_FS_MAKE_SOCK = 1 << 9
ACCESS_FS_MAKE_FIFO = 1 << 10
ACCESS_FS_MAKE_BLOCK = 1 << 11
ACCESS_FS_MAKE_SYM = 1 << 12
ACCESS_FS_REFER = 1 << 13     # ABI >= 2 (kernel 5.19)
ACCESS_FS_TRUNCATE = 1 << 14  # ABI >= 3 (kernel 6.2)

# every write-side access right; read/execute rights are left unhandled so they stay allowed everywhere
WRITE_ACCESS = (ACCESS_FS_WRITE_FILE | ACCESS_FS_REMOVE_DIR | ACCESS_FS_REMOVE_FILE |
                ACCESS_FS_MAKE_CHAR | ACCESS_FS_MAKE_DIR | ACCESS_FS_MAKE_REG | ACCESS_FS_MAKE_SOCK |
                ACCESS_FS_MAKE_FIFO | ACCESS_FS_MAKE_BLOCK | ACCESS_FS_MAKE_SYM |
                ACCESS_FS_REFER | ACCESS_FS_TRUNCATE)
# rights that apply to regular files (the rest are directory-only and rejected by the kernel on file rules)
FILE_ACCESS = ACCESS_FS_WRITE_FILE | ACCESS_FS_TRUNCATE

SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
PR_SET_NO_NEW_PRIVS = 38


class LandlockRulesetAttr(ctypes.Structure):
  _fields_ = (("handled_access_fs", ctypes.c_uint64),)


class LandlockPathBeneathAttr(ctypes.Structure):
  _pack_ = 1  # packed in the uapi header
  _fields_ = (("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32))


_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def _syscall(nr: int, *args) -> int:
  res = _libc.syscall(ctypes.c_long(nr), *args)
  if res < 0:
    err = ctypes.get_errno()
    raise OSError(err, os.strerror(err))
  return res


def landlock_abi_version() -> int:
  """Highest Landlock ABI version supported, or 0 if the kernel does not support Landlock."""
  try:
    return _syscall(SYS_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION))
  except OSError:
    return 0


def _add_rule(ruleset_fd: int, path: str, allowed_access: int):
  fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
  try:
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
      allowed_access &= FILE_ACCESS
    attr = LandlockPathBeneathAttr(allowed_access=allowed_access, parent_fd=fd)
    _syscall(SYS_LANDLOCK_ADD_RULE, ctypes.c_int(ruleset_fd), ctypes.c_uint32(LANDLOCK_RULE_PATH_BENEATH), ctypes.byref(attr), ctypes.c_uint32(0))
  finally:
    os.close(fd)


def restrict_file_writes(read_write_paths: Iterable[str], write_file_paths: Iterable[str] = ()) -> bool:
  """Deny filesystem writes for this process (and all future children) outside the given paths.

  read_write_paths: full write access (create/modify/delete) beneath these paths.
  write_file_paths: existing files beneath these paths may only be opened for writing,
                    nothing can be created or removed (e.g. "/dev" for device nodes).

  Reads and execution stay unrestricted. Whitelist paths that don't exist are skipped.
  Also blocks filesystem topology changes (mount/umount) and, via NO_NEW_PRIVS, setuid binaries.
  Returns False if the kernel does not support Landlock, True once the sandbox is applied.
  """
  abi = landlock_abi_version()
  if abi < 1:
    return False

  handled = WRITE_ACCESS
  if abi < 2:
    handled &= ~ACCESS_FS_REFER
  if abi < 3:
    handled &= ~ACCESS_FS_TRUNCATE

  ruleset_attr = LandlockRulesetAttr(handled_access_fs=handled)
  ruleset_fd = _syscall(SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(ruleset_attr), ctypes.c_size_t(ctypes.sizeof(ruleset_attr)), ctypes.c_uint32(0))
  try:
    for paths, access in ((read_write_paths, handled), (write_file_paths, ACCESS_FS_WRITE_FILE)):
      for path in paths:
        try:
          _add_rule(ruleset_fd, path, access)
        except FileNotFoundError:
          pass

    # landlock_restrict_self requires no_new_privs; this also applies to the calling thread only,
    # so the sandbox must be installed before any threads are spawned
    if _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
      err = ctypes.get_errno()
      raise OSError(err, os.strerror(err))
    _syscall(SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset_fd), ctypes.c_uint32(0))
  finally:
    os.close(ruleset_fd)
  return True
