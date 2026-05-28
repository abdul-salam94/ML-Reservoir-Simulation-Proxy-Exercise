#!/bin/bash
# Migrate preprocessed dataset from /mnt/d (slow 9p) to /root (fast ext4)
# for ~5-10x training I/O speedup.

set -e

# 1. Kill any running training
if pgrep -f "train.py --config-name=config_norne_pipeline_test" > /dev/null; then
    echo "==> Killing current training process..."
    pkill -f "train.py --config-name=config_norne_pipeline_test"
    sleep 2
fi

# 2. Make placeholder sim_dir + dataset target
mkdir -p /root/NORNE/completed
DST=/root/NORNE/completed.dataset/XMGN_Norne_PipelineTest_60
mkdir -p "$DST"

SRC=/mnt/d/NORNE/completed.dataset/XMGN_Norne_PipelineTest_60

# 3. Copy partition .pt files (the big I/O for training) — about 20 GB
echo "==> Copying partitions/ (3,968 partition files, ~20 GB)..."
time cp -r "$SRC/partitions" "$DST/"

# 4. Copy the small metadata files
echo "==> Copying JSON metadata..."
cp "$SRC/global_stats.json"             "$DST/"
cp "$SRC/dataset_metadata.json"         "$DST/"
cp "$SRC/preprocessing_metadata.json"   "$DST/" 2>/dev/null || true
cp "$SRC/"NORNE_*_partitions.json       "$DST/" 2>/dev/null || true

# 5. Verify
echo ""
echo "==> Verification:"
echo "  Train partitions:   $(ls $DST/partitions/train | wc -l)"
echo "  Val partitions:     $(ls $DST/partitions/val | wc -l)"
echo "  Test partitions:    $(ls $DST/partitions/test | wc -l)"
echo "  global_stats.json:  $(ls -lh $DST/global_stats.json | awk '{print $5}')"
echo "  Total dataset size: $(du -sh $DST | awk '{print $1}')"
echo ""
echo "==> Done. New training data lives at $DST"
echo "==> Set config sim_dir to: /root/NORNE/completed"
