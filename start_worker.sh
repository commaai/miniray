#!/bin/bash -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${SCRIPT_DIR}

export TRITON_SERVER_ENABLED=${TRITON_SERVER_ENABLED:-1}

echo "start worker ..."
export UV_CACHE_DIR=$HOME/.cache/uv
export UV_PYTHON_INSTALL_DIR=$HOME/.local/share/uv/python
sudo \
  --preserve-env=UV_CACHE_DIR \
  --preserve-env=UV_PYTHON_INSTALL_DIR \
  --preserve-env=TASK_UID \
  --preserve-env=DEBUG_WORKER \
  --preserve-env=REDIS_HOST \
  --preserve-env=PIPELINE_QUEUE \
  --preserve-env=CPU_COUNT \
  --preserve-env=GPU_COUNT \
  --preserve-env=FORCE_SMALL_GPU \
  --preserve-env=TRITON_SERVER_ENABLED \
  bash -c "PATH=$HOME/.local/bin:\$PATH && uv run worker.py"
