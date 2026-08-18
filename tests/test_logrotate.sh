#!/usr/bin/env bash
set -e
# setup_isolation.sh installs a logrotate config to cap worker log size; validate it parses
logrotate -d /etc/logrotate.d/miniray-worker 2>&1 | grep -q "rotating pattern: /var/log/slurm/miniray-worker"
