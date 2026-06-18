"""
分析為什麼 CFG 在 AFHQ 上反而降低品質

這個腳本用於報告的 Ablation Study 部分
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 從 results.json 讀取數據
with open('results/eval_task2/results.json', 'r') as f:
    results = json.load(f)

# 整理數據
cfg_scales = [1.0, 3.0, 7.5]
steps_list = [10, 20, 50]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: FID vs CFG Scale (固定 steps=50)
fids_at_50_steps = []
for cfg in cfg_scales:
    key = f"cfg{cfg}_steps50"
    fids_at_50_steps.append(results[key]['fid'])

axes[0].plot(cfg_scales, fids_at_50_steps, 'o-', linewidth=2, markersize=8)
axes[0].axhline(y=30, color='r', linestyle='--', label='Target FID=30', alpha=0.5)
axes[0].set_xlabel('CFG Scale', fontsize=12)
axes[0].set_ylabel('FID Score', fontsize=12)
axes[0].set_title('FID vs CFG Scale (50 Steps)\n⚠️ Higher CFG → Worse FID', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# 添加數值標註
for i, (cfg, fid) in enumerate(zip(cfg_scales, fids_at_50_steps)):
    axes[0].text(cfg, fid + 0.5, f'{fid:.2f}', ha='center', fontsize=10)

# Plot 2: FID vs Inference Steps (不同 CFG)
for cfg in cfg_scales:
    fids = []
    for steps in steps_list:
        key = f"cfg{cfg}_steps{steps}"
        fids.append(results[key]['fid'])
    axes[1].plot(steps_list, fids, 'o-', linewidth=2, markersize=8, label=f'CFG={cfg}')

axes[1].axhline(y=30, color='r', linestyle='--', label='Target FID=30', alpha=0.5)
axes[1].set_xlabel('Inference Steps', fontsize=12)
axes[1].set_ylabel('FID Score', fontsize=12)
axes[1].set_title('FID vs Inference Steps\n✅ CFG=1.0 Always Best', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('results/eval_task2/cfg_failure_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Saved analysis plot to results/eval_task2/cfg_failure_analysis.png")

# 生成文字報告
print("\n" + "="*60)
print("📊 CFG Failure Analysis for AFHQ Dataset")
print("="*60)

print("\n🔍 Key Findings:")
print(f"1. Best FID: {min(fids_at_50_steps):.2f} at CFG=1.0 (no guidance)")
print(f"2. Worst FID: {max(fids_at_50_steps):.2f} at CFG=7.5 (strong guidance)")
print(f"3. FID degradation: {((max(fids_at_50_steps) - min(fids_at_50_steps)) / min(fids_at_50_steps) * 100):.1f}% increase")

print("\n❌ Why CFG Fails Here:")
print("   • cfg_dropout=0.1 too low (only 10% unconditional training)")
print("   • AFHQ classes (cat/dog/wild) don't strongly affect quality")
print("   • FID measures diversity across all classes, not per-class quality")
print("   • High CFG → mode collapse → reduced diversity → higher FID")

print("\n✅ When CFG Works (theory):")
print("   • cfg_dropout ≥ 0.3 (30%+ unconditional training)")
print("   • Strong semantic conditioning (e.g., text-to-image)")
print("   • Task requires high prompt fidelity over diversity")

print("\n💡 Solutions:")
print("   1. Retrain with cfg_dropout=0.5")
print("   2. Use CFG=1.0~2.0 for AFHQ (treat as unconditional)")
print("   3. Consider unconditional training (no --use_cfg)")

print("\n" + "="*60)

# 生成表格數據（用於報告）
print("\n📋 Results Table (for report):")
print("\n| CFG Scale | Steps 10 | Steps 20 | Steps 50 |")
print("|-----------|----------|----------|----------|")
for cfg in cfg_scales:
    row = f"| {cfg:3.1f}     |"
    for steps in steps_list:
        key = f"cfg{cfg}_steps{steps}"
        row += f" {results[key]['fid']:6.2f}   |"
    print(row)

print("\n" + "="*60)
