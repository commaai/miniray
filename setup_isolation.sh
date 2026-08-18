#!/bin/bash -e

cat > /etc/logrotate.d/miniray-worker <<'EOF'
/var/log/slurm/miniray-worker.*.log {
  size 1G
  copytruncate
}
EOF

for PROC_ID in $(seq -f '%03g' 0 $(($(nproc) - 1))); do
  # return code 9 = user already exists
  useradd --no-create-home --shell /usr/sbin/nologin --home /nonexistent --system proc${PROC_ID} 2> /dev/null || [ $? -eq 9 ]
done

groupadd -f video && groupadd -f docker
