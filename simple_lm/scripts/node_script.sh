#!/bin/bash
echo "Hostname: $(hostname -s)"
echo "Node Rank ${SLURM_PROCID}"
# prepare environment
source /h/mchoi/envs/simple_lm_env/bin/activate
echo "Using Python from: $(which python)"

python3.9 slurm_example_gtc.py --coordinator_address "${MASTER_ADDR}:${MASTER_PORT}" --num_processes 2
