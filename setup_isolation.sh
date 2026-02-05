#!/bin/bash -e
for PROC_ID in $(seq -f '%03g' 0 $(($(nproc) - 1))); do
  # return code 9 = user already exists
  useradd --no-create-home --shell /usr/sbin/nologin --home /nonexistent --system proc${PROC_ID} 2> /dev/null || [ $? -eq 9 ]
done
