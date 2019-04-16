#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Usage: ./run_multiple_workers.sh [num_workers] [ip_addr] [worker_type] [logdir]"
  exit
fi

let port=60061
let max_id=$1-1
for i in $(seq 0 $max_id) 
do
  python3 worker.py -i $2 -t $3 -g $i -w $port > $4/worker_$i.log 2>&1 &
done
