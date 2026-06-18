#!/usr/bin/env python3
"""
Evaluate Task 2 model (Flow Matching) across different CFG scales and inference steps.

Usage (run inside FlowMatching conda env):
    python scripts/eval_task2_cfg_steps.py \
        --ckpt results/cfg_fm-11-17-125702/last.ckpt \
        --num_samples 500 \
        --cfg_scales 1.0,3.0,7.5 \
        --steps 10,20,50

This script will:
 - run sampling for each (cfg_scale, num_steps) combo and save images to results/eval_task2/<combo>
 - compute FID against data/afhq/eval using fid.measure_fid
 - create a small grid image of samples for quick visual comparison
 - write results to results/eval_task2/results.json

It uses the project's existing sampling and fid modules via subprocess calls.
"""
import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
from PIL import Image


def run_cmd(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode, proc.stdout


def make_grid(img_dir, out_path, grid_size=(8,8), ext=('png','jpg','jpeg')):
    files = sorted([p for p in pathlib.Path(img_dir).iterdir() if p.suffix.lower().lstrip('.') in ext])
    if len(files) == 0:
        return False
    files = files[: grid_size[0]*grid_size[1] ]
    imgs = [Image.open(f).convert('RGB') for f in files]
    w,h = imgs[0].size
    grid_w = grid_size[0]*w
    grid_h = grid_size[1]*h
    grid = Image.new('RGB', (grid_w, grid_h), (255,255,255))
    for idx, im in enumerate(imgs):
        x = (idx % grid_size[0]) * w
        y = (idx // grid_size[0]) * h
        grid.paste(im, (x,y))
    grid.save(out_path)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='Task2 checkpoint path')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--cfg_scales', default='1.0,3.0,7.5', help='Comma separated CFG scales')
    parser.add_argument('--steps', default='10,20,50', help='Comma separated inference steps')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_root', default='results/eval_task2')
    parser.add_argument('--eval_ref', default='data/afhq/eval')
    args = parser.parse_args()

    cfg_scales = [float(x) for x in args.cfg_scales.split(',')]
    steps_list = [int(x) for x in args.steps.split(',')]

    os.makedirs(args.save_root, exist_ok=True)

    results = {}

    for scale in cfg_scales:
        for steps in steps_list:
            combo_name = f'cfg{scale}_steps{steps}'
            out_dir = os.path.join(args.save_root, combo_name)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir, exist_ok=True)

            # Run sampling (sampling.py doesn't have --num_samples, it generates 500 by default)
            samp_cmd = [
                'python', '-m', 'image_common.sampling',
                '--use_cfg',
                '--ckpt_path', args.ckpt,
                '--save_dir', out_dir,
                '--num_inference_steps', str(steps),
                '--cfg_scale', str(scale)
            ]
            rc, out = run_cmd(samp_cmd)
            if rc != 0:
                results[combo_name] = {'error': 'sampling failed', 'output': out}
                continue

            # Build grid
            grid_path = os.path.join(out_dir, 'grid.png')
            grid_ok = make_grid(out_dir, grid_path, grid_size=(8,8))

            # Compute FID
            fid_cmd = ['python', '-m', 'fid.measure_fid', args.eval_ref, out_dir]
            rc, out = run_cmd(fid_cmd)
            fid_value = None
            for line in out.splitlines():
                if 'FID:' in line:
                    try:
                        fid_value = float(line.strip().split('FID:')[-1])
                    except Exception:
                        pass

            results[combo_name] = {
                'cfg_scale': scale,
                'steps': steps,
                'fid': fid_value,
                'grid': grid_path if grid_ok else None,
                'samples_dir': out_dir
            }

            # write intermediate results
            with open(os.path.join(args.save_root, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)

    print('All done. Results saved to', os.path.join(args.save_root, 'results.json'))


if __name__ == '__main__':
    main()
