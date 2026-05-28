#!/bin/bash
# Launch xmgn training fully detached so it survives PowerShell/WSL session changes.
# Output goes to /root/train.log inside WSL.

# Distributed env vars for single-process training
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export LOCAL_WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

cd /mnt/e/NVIDIA/reservoir_simulation/xmgn
nohup /root/xmgn-env/bin/python -u src/train.py --config-name=config_norne_pipeline_test \
    > /root/train.log 2>&1 < /dev/null &
PID=$!
disown
echo "Launched training as PID $PID"
echo "Log: /root/train.log"
sleep 5
echo "--- first 15 log lines ---"
head -15 /root/train.log 2>/dev/null || echo "(no output yet)"
echo "--- is PID alive? ---"
ps -p $PID > /dev/null && echo "  YES, training running (PID $PID)" || echo "  NO, training exited"
