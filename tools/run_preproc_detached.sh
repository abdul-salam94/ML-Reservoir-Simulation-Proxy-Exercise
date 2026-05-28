#!/bin/bash
# Run preprocessor fully detached so it survives PowerShell session changes.
# Output goes to /root/preprocess.log inside WSL.

cd /mnt/e/NVIDIA/reservoir_simulation/xmgn
nohup /root/xmgn-env/bin/python -u src/preprocessor.py --config-name=config_norne_pipeline_test \
    > /root/preprocess.log 2>&1 < /dev/null &
PID=$!
disown
echo "Launched preprocessor as PID $PID"
echo "Log: /root/preprocess.log"
sleep 3
echo "--- first lines of log ---"
head -10 /root/preprocess.log 2>/dev/null || echo "(no output yet)"
echo "--- is PID alive? ---"
ps -p $PID > /dev/null && echo "  YES, still running (PID $PID)" || echo "  NO, process exited"
