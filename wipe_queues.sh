#!/bin/bash
identifier="$1"

if [[ ${#identifier} == 0 ]]
then
  echo "please specify task name. (e.g. wipe_queues.sh task_name)"
  exit
fi

echo "-- tasks found for $identifier*: --"
for key in $(redis-cli -h redis.comma.internal -n 1 keys '*')
do
  if [[ $key == $identifier* ]]
  then
    echo $key $(redis-cli -h redis.comma.internal -n 1 llen $key)
  fi
done

read -r -p "-- Are you sure to clear them? [y/N] --" response
case "$response" in
  [yY]|[yY])
    for key in $(redis-cli -h redis.comma.internal -n 1 keys '*')
    do
      if [[ $key == $identifier* ]]
      then
        redis-cli -h redis.comma.internal -n 1 del $key
        redis-cli -h redis.comma.internal -n 1 del "tasks:$key" &>/dev/null
        redis-cli -h redis.comma.internal -n 1 del "job-metadata:$key" &>/dev/null
      fi
    done
    ;;
  *)
    exit
    ;;
esac
