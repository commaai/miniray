#!/bin/bash -e
REDIS_CMD="redis-cli -h redis.comma.internal -n 1"

for KEY in $(${REDIS_CMD} keys 'block:*')
do
  if [[ $KEY == *$QUEUE* ]]
  then
    echo "${KEY#*:} : $(${REDIS_CMD} ttl $KEY) seconds"
  fi
done
