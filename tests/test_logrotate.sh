#!/usr/bin/env bash
set -e
# just confirm it parses
logrotate -d /etc/logrotate.d/miniray-worker 2>&1 | grep -q "rotating pattern: /var/log/slurm/miniray-worker"
