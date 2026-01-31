#!/bin/bash -e
REDIS_CMD="redis-cli -h redis.comma.internal -n 1"
IFS=$'\n'
KEYS=($(eval "$REDIS_CMD keys 'suspend:*' | sort"))
if [ "${#KEYS[@]}" -eq "0" ]; then
    exit 0
fi
VALS=($(eval "$REDIS_CMD mget ${KEYS[@]}"))
if [ "${#KEYS[@]}" -ne "${#VALS[@]}" ]; then
    echo "arrays are not the same length"
    exit 1
fi
echo "suspended workers:"
for i in ${!KEYS[*]}; do
  printf "%-20s %s\n" ${KEYS[$i]#*:} ${VALS[$i]}
done
