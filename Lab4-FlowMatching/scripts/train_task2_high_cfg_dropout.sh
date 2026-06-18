#!/bin/bash

# 重新訓練 Task 2 with cfg_dropout=0.5
# 這個版本應該能讓 CFG 正常工作

cd /home/c0922/IVG/Lab4-FlowMatching
conda activate FlowMatching

python -m task2_image_flow_matching.train \
    --use_cfg \
    --cfg_dropout 0.5 \
    --train_num_steps 100000 \
    --batch_size 16 \
    --lr 2e-4

echo "Training complete! Now evaluate with different CFG scales:"
echo "python scripts/eval_task2_cfg_steps.py --ckpt results/cfg_fm-<new_timestamp>/last.ckpt --cfg_scales 1.0,3.0,7.5 --steps 50"
