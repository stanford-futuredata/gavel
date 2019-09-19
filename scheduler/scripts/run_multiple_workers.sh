#!/bin/bash

if [ "$#" -ne 6 ]; then
  echo "Usage: ./run_multiple_workers.sh [num_workers] [ip_addr] [worker_type] [logdir] [time_per_iteration] [min_gpu_id]"
  exit
fi

let port=60061
let max_id=$1-1
let min_id=$6
for i in $(seq $min_id $max_id)
do
  python worker.py -i $2 -t $3 -g $i -w $port --time_per_iteration $5 > $4/worker_$i.log 2>&1 &
  let port=$port+1
done
