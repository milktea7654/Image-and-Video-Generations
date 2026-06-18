#!/usr/bin/env python3
import torch
import sys
import os

# 暫時修改 torch.load 的預設行為來避免權重檢查問題
original_load = torch.load
def patched_load(f, map_location=None, pickle_module=None, **kwargs):
    return original_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False)
torch.load = patched_load

# 設定命令列參數
sys.argv = [
    'sampling.py',
    '--ckpt_path', '/home/c0922/DDPM/Lab1-DDPM/image_diffusion_todo/results/predictor_mean/beta_linear/09-28-182805/last.ckpt',
    '--save_dir', '/home/c0922/DDPM/Lab1-DDPM/image_diffusion_todo/results/predictor_mean/beta_linear/09-28-182805/sample',
    '--mode', 'linear',
    '--predictor', 'mean'
]

# 執行原始的 sampling.py
exec(open('sampling.py').read())