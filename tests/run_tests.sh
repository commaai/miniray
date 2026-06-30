#!/usr/bin/env bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/.."

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-miniray-test-$$}"

cleanup() {
  echo "[CLEANUP] Showing worker logs..."
  docker compose -f docker-compose.ci.yml logs worker || true
  echo "[CLEANUP] Stopping containers..."
  docker compose -f docker-compose.ci.yml down -v --remove-orphans || true
}

trap cleanup EXIT

docker compose -f docker-compose.ci.yml build
docker compose -f docker-compose.ci.yml up -d redis worker
docker compose -f docker-compose.ci.yml run --rm test bash -c "
  sleep 5 &&
  cd /app/miniray/ &&
  ruff check . &&
  ty check . &&
  python3 -m pytest -n12 -v -m 'not dstate' /app/miniray/tests/ &&
  python3 -m pytest -v -m dstate /app/miniray/tests/test_miniray.py
"
