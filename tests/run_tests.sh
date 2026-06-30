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
docker compose -f docker-compose.ci.yml up --exit-code-from test --abort-on-container-exit
