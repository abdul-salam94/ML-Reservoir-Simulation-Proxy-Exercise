#!/bin/bash
# Fix the metadata paths and restart training. Uses PID-targeted kill to
# avoid pkill matching this script itself.

META=/root/NORNE/completed.dataset/XMGN_Norne_PipelineTest_60/dataset_metadata.json

# 1. Kill running xmgn training by python interpreter path (not script name)
echo "=== Kill running training ==="
for PID in $(ps -eo pid,cmd | grep '/root/xmgn-env/bin/python' | grep config_norne_pipeline_test | grep -v grep | awk '{print $1}'); do
    echo "  Killing PID $PID"
    kill -TERM "$PID"
done
sleep 3

# 2. Rewrite paths
echo
echo "=== Rewrite paths in metadata ==="
echo "BEFORE:"
grep '"dir"\|"_dir"\|"file"' "$META" 2>/dev/null
sed -i 's|/mnt/d/NORNE|/root/NORNE|g' "$META"
echo "AFTER:"
grep '"dir"\|"_dir"\|"file"' "$META"

# 3. Re-launch training (using the existing detached launcher)
echo
echo "=== Re-launch training ==="
bash /mnt/e/NVIDIA/reservoir_simulation/tools/run_train_detached.sh
