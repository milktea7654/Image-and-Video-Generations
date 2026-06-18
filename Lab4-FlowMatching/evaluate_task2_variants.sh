#!/bin/bash
# Task 2 評估腳本：測試不同的 CFG scales 和 inference steps

cd /home/c0922/IVG/Lab4-FlowMatching
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FlowMatching

CKPT_PATH="results/cfg_fm-11-17-125702/last.ckpt"
EVAL_DATA="data/afhq/eval"

echo "=================================="
echo "Task 2: Flow Matching Evaluation"
echo "Testing different CFG scales and inference steps"
echo "=================================="
echo ""

# 測試不同的 CFG scales（固定 20 steps）
echo "===== Testing CFG Scales (20 steps) ====="
for cfg_scale in 1.0 3.0 7.5; do
    save_dir="results/task2_cfg${cfg_scale}_steps20"
    echo ""
    echo "▶ CFG Scale: ${cfg_scale}, Steps: 20"
    echo "  Generating samples..."
    
    python -m image_common.sampling \
        --use_cfg \
        --ckpt_path ${CKPT_PATH} \
        --save_dir ${save_dir} \
        --num_inference_steps 20 \
        --cfg_scale ${cfg_scale}
    
    if [ $? -eq 0 ]; then
        echo "  Calculating FID..."
        fid=$(python -m fid.measure_fid ${EVAL_DATA} ${save_dir} 2>&1 | grep "FID:" | awk '{print $2}')
        echo "  ✓ FID = ${fid}"
        echo "CFG=${cfg_scale}, Steps=20, FID=${fid}" >> task2_results.txt
    else
        echo "  ✗ Generation failed"
    fi
done

echo ""
echo "===== Testing Inference Steps (CFG=7.5) ====="
# 測試不同的 inference steps（固定 CFG=7.5）
for steps in 10 20 50; do
    save_dir="results/task2_cfg7.5_steps${steps}"
    echo ""
    echo "▶ CFG Scale: 7.5, Steps: ${steps}"
    echo "  Generating samples..."
    
    python -m image_common.sampling \
        --use_cfg \
        --ckpt_path ${CKPT_PATH} \
        --save_dir ${save_dir} \
        --num_inference_steps ${steps} \
        --cfg_scale 7.5
    
    if [ $? -eq 0 ]; then
        echo "  Calculating FID..."
        fid=$(python -m fid.measure_fid ${EVAL_DATA} ${save_dir} 2>&1 | grep "FID:" | awk '{print $2}')
        echo "  ✓ FID = ${fid}"
        echo "CFG=7.5, Steps=${steps}, FID=${fid}" >> task2_results.txt
    else
        echo "  ✗ Generation failed"
    fi
done

echo ""
echo "=================================="
echo "All evaluations completed!"
echo "Results saved to task2_results.txt"
echo "=================================="
echo ""
echo "Summary:"
cat task2_results.txt
