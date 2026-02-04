#!/usr/bin/env bash
set -ex

# Run miniray CI tests using docker compose
# This script mirrors what pipeline/miniray_worker_tests/test.sh miniray does
# but is self-contained for the miniray submodule

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MINIRAY_ROOT="$DIR/.."

cd "$MINIRAY_ROOT"

export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-miniray-test-$$}"

cleanup() {
  echo "[CLEANUP] Showing worker logs..."
  docker compose -f docker-compose.ci.yml logs worker || true
  echo "[CLEANUP] Stopping containers..."
  docker compose -f docker-compose.ci.yml down -v --remove-orphans || true
}

trap cleanup EXIT

echo "[BUILD] Building docker images..."
docker compose -f docker-compose.ci.yml build

echo "[START] Starting services..."
docker compose -f docker-compose.ci.yml up --exit-code-from test --abort-on-container-exit

echo "[DONE] Tests completed successfully!"
