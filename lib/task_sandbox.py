from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

SANDBOX_ENV = "MINIRAY_SANDBOX"
WRITE_PATHS_ENV = "MINIRAY_SANDBOX_WRITE_PATHS"

LANDLOCK_BACKEND = "landlock"


def sandbox_backends(env: Mapping[str, str] = os.environ) -> frozenset[str]:
  spec = env.get(SANDBOX_ENV, LANDLOCK_BACKEND).strip().lower()
  if spec in {"", "1", "true", "yes", "on"}:
    return frozenset({LANDLOCK_BACKEND})
  if spec in {"0", "false", "no", "off", "none", "disabled"}:
    return frozenset()

  backends = frozenset(part.strip() for part in spec.replace("+", ",").split(",") if part.strip())
  invalid = backends - {LANDLOCK_BACKEND}
  if invalid:
    raise ValueError(f"unsupported {SANDBOX_ENV} backend(s): {', '.join(sorted(invalid))}")
  return backends


def sandbox_uses_landlock(env: Mapping[str, str] = os.environ) -> bool:
  return LANDLOCK_BACKEND in sandbox_backends(env)


def sandbox_write_paths(env: Mapping[str, str] = os.environ, *, create_pycache: bool = False, existing_only: bool = False) -> list[str]:
  pycache_dir = Path(env["PYTHONPYCACHEPREFIX"])
  if create_pycache:
    pycache_dir.mkdir(parents=True, exist_ok=True)

  paths = [
    env["TMPDIR"],
    env["CUPY_CACHE_DIR"],
    str(pycache_dir),
    "/dev/shm",
    *filter(None, env.get(WRITE_PATHS_ENV, "").split(":")),
  ]

  deduped = []
  seen = set()
  for path in paths:
    normalized = str(Path(path))
    if normalized in seen or (existing_only and not Path(normalized).exists()):
      continue
    seen.add(normalized)
    deduped.append(normalized)
  return deduped
