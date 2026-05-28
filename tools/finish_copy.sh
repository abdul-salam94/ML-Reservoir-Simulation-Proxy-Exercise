#!/bin/bash
# Finish what the interrupted copy didn't: tiny JSON files + cleanup of partial top-level copies.

DST=/root/NORNE/completed.dataset/XMGN_Norne_PipelineTest_60
SRC=/mnt/d/NORNE/completed.dataset/XMGN_Norne_PipelineTest_60

echo "=== Files in destination root before fix ==="
ls -lh "$DST/" | head -20
echo

echo "=== Copy missing essential JSONs ==="
cp -v "$SRC/global_stats.json"           "$DST/"
cp -v "$SRC/dataset_metadata.json"       "$DST/"
cp -v "$SRC/preprocessing_metadata.json" "$DST/" 2>/dev/null || echo "(preprocessing_metadata.json not in source, skipping)"
echo

echo "=== Copy NORNE_NNN_partitions.json per-case files ==="
cp -v "$SRC/"NORNE_*_partitions.json "$DST/" 2>/dev/null | tail -3
echo "(per-case partition JSONs copied: $(ls $DST/NORNE_*_partitions.json 2>/dev/null | wc -l))"
echo

echo "=== Delete partial top-level partition .pt files ==="
N_BEFORE=$(find "$DST/partitions" -maxdepth 1 -name "*.pt" | wc -l)
find "$DST/partitions" -maxdepth 1 -name "*.pt" -delete
N_AFTER=$(find "$DST/partitions" -maxdepth 1 -name "*.pt" | wc -l)
echo "  Removed $N_BEFORE - $N_AFTER = $((N_BEFORE - N_AFTER)) partial files"
echo

echo "=== Final state ==="
echo "Dataset size:   $(du -sh $DST | awk '{print $1}')"
echo "WSL vdisk free: $(df -h / | tail -1 | awk '{print $4}')"
echo "Train .pt:      $(ls $DST/partitions/train | wc -l)"
echo "Val .pt:        $(ls $DST/partitions/val | wc -l)"
echo "Test .pt:       $(ls $DST/partitions/test | wc -l)"
echo "global_stats:   $(ls -lh $DST/global_stats.json 2>/dev/null | awk '{print $5}')"
echo "dataset_meta:   $(ls -lh $DST/dataset_metadata.json 2>/dev/null | awk '{print $5}')"
