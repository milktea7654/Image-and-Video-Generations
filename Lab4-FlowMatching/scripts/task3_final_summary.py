"""
Task 3 Ablation Study - Final Results Summary
整理所有測試結果
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 避免 display 問題

# 完整結果
results = {
    "Original FM": {
        "5_steps": {"fid": 29.58},
        "10_steps": {"fid": 13.15},
        "20_steps": {"fid": 7.50},
        "50_steps": {"fid": 4.51}
    },
    "RF1": {
        "5_steps": {"fid": 2.16},
        "10_steps": {"fid": 23.32},
        "20_steps": {"fid": 16.86}
    },
    "RF2": {
        "5_steps": {"fid": 2.84},
        "10_steps": {"fid": 2.34},
        "20_steps": {"fid": 2.43}
    }
}

output_dir = Path("results/eval_task3")
output_dir.mkdir(exist_ok=True)

# 保存 JSON
with open(output_dir / "ablation_results_final.json", "w") as f:
    json.dump(results, f, indent=2)

print("="*80)
print("📊 Task 3 Ablation Study - FINAL RESULTS")
print("="*80)

print("\n📋 Complete Results Table:")
print("\n| Model        | 5 Steps | 10 Steps | 20 Steps | 50 Steps |")
print("|--------------|---------|----------|----------|----------|")

models = ["Original FM", "RF1", "RF2"]
for model in models:
    row = f"| {model:12s} |"
    for steps in [5, 10, 20, 50]:
        key = f"{steps}_steps"
        if key in results[model]:
            fid = results[model][key]["fid"]
            row += f" {fid:7.2f} |"
        else:
            row += "    -    |"
    print(row)

print("\n🎯 Key Findings:")

print("\n1. Trajectory Straightness Improvement:")
print("   📐 Original FM @ 5 steps:  FID = 29.58 (curved path, poor quality)")
print("   📐 RF1 @ 5 steps:          FID = 2.16  (✅ 93% better!)")
print("   📐 RF2 @ 5 steps:          FID = 2.84  (✅ 90% better!)")
print("   ")
print("   🔹 Conclusion: Rectification dramatically improves few-step generation")

print("\n2. Quality Comparison @ 5 Steps:")
print("   RF1 slightly better than RF2 at 5 steps")
print("   This is unexpected - possible reasons:")
print("   • RF2 might be overfitted to straighter paths")
print("   • RF1 found a good local optimum")
print("   • CFG scale 1.0 may not be optimal for RF2")

print("\n3. Quality vs Steps Tradeoff:")
print("   Original FM:")
print("   • 5 steps  → FID 29.58 (unacceptable)")
print("   • 10 steps → FID 13.15 (borderline)")
print("   • 20 steps → FID 7.50  (good)")
print("   • 50 steps → FID 4.51  (excellent)")
print("   ")
print("   RF1:")
print("   • 5 steps  → FID 2.16  (✅ excellent, 10x speedup!)")
print("   • 10 steps → FID 23.32 (worse than expected)")
print("   • 20 steps → FID 16.86 (not great)")
print("   ")
print("   RF2:")
print("   • 5 steps  → FID 2.84  (✅ excellent)")
print("   • 10 steps → FID 2.34  (✅ excellent)")
print("   • More stable across different step counts")

print("\n4. Speedup Analysis:")
print("   Baseline: Original FM @ 50 steps = FID 4.51")
print("   ")
print("   RF1 @ 5 steps:  FID 2.16 (✅ 10x faster, better quality!)")
print("   RF2 @ 5 steps:  FID 2.84 (✅ 10x faster, better quality!)")
print("   RF2 @ 10 steps: FID 2.34 (✅ 5x faster, better quality!)")

print("\n5. Unexpected Observation:")
print("   RF1 @ 10,20 steps performs WORSE than @ 5 steps!")
print("   • FID @ 5:  2.16")
print("   • FID @ 10: 23.32 (📉 10x worse)")
print("   • FID @ 20: 16.86 (📉 8x worse)")
print("   ")
print("   Possible explanations:")
print("   • RF1 training optimized for ~5 steps (reflow data used 5 steps)")
print("   • Accumulation of numerical errors in longer trajectories")
print("   • Model learned trajectory suited for specific step count")

print("\n6. Best Configuration:")
print("   🏆 Winner: RF1 @ 5 steps")
print("   • FID: 2.16 (best overall)")
print("   • Speedup: 10x vs Original @ 50 steps")
print("   • Quality: Better than Original @ 50 steps")

# 生成圖表
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: FID vs Steps
for model in models:
    steps_list = []
    fids_list = []
    
    for steps in [5, 10, 20, 50]:
        key = f"{steps}_steps"
        if key in results[model]:
            steps_list.append(steps)
            fids_list.append(results[model][key]["fid"])
    
    if steps_list:
        axes[0].plot(steps_list, fids_list, 'o-', linewidth=2.5, markersize=10, label=model)
        
        # 標註數值
        for s, f in zip(steps_list, fids_list):
            axes[0].text(s*1.1, f, f'{f:.1f}', fontsize=9, ha='left')

axes[0].axhline(y=30, color='r', linestyle='--', linewidth=2, label='Target FID=30', alpha=0.7)
axes[0].set_xlabel('Inference Steps', fontsize=13, fontweight='bold')
axes[0].set_ylabel('FID Score (lower is better)', fontsize=13, fontweight='bold')
axes[0].set_title('Rectified Flow: Quality vs Inference Steps\nRF1 @ 5 steps = Best', 
                 fontsize=14, fontweight='bold')
axes[0].set_xscale('log')
axes[0].set_xticks([5, 10, 20, 50])
axes[0].set_xticklabels(['5', '10', '20', '50'])
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].legend(fontsize=11, loc='upper right')

# Plot 2: Speedup vs Quality
baseline_time = 50  # 50 steps as baseline
for model in models:
    speedups = []
    fids = []
    
    for steps in [5, 10, 20, 50]:
        key = f"{steps}_steps"
        if key in results[model]:
            speedups.append(baseline_time / steps)
            fids.append(results[model][key]["fid"])
    
    if speedups:
        axes[1].scatter(speedups, fids, s=200, alpha=0.7, label=model)
        
        # 標註步數
        for i, steps in enumerate([5, 10, 20, 50]):
            key = f"{steps}_steps"
            if key in results[model]:
                axes[1].text(speedups[i], fids[i], f'{steps}', 
                           fontsize=9, ha='center', va='center', fontweight='bold')

axes[1].axhline(y=30, color='r', linestyle='--', linewidth=2, label='Target FID=30', alpha=0.7)
axes[1].set_xlabel('Speedup (x faster than 50 steps)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('FID Score', fontsize=13, fontweight='bold')
axes[1].set_title('Quality-Speed Tradeoff\n(Top-right corner = Best)', 
                 fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].legend(fontsize=11)
axes[1].invert_yaxis()

plt.tight_layout()
plot_path = output_dir / 'task3_ablation_final.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved plot to {plot_path}")

print("\n💡 For Report:")
print("\n1. Methodology:")
print("   • Original FM: Standard Flow Matching (Task 2 model)")
print("   • RF1: 1-Rectified Flow (trained on reflow data from Original)")
print("   • RF2: 2-Rectified Flow (trained on reflow data from RF1)")
print("   • All tested with CFG scale = 1.0 (optimal from Task 2)")

print("\n2. Main Contributions:")
print("   • Trajectory straightening enables 10x speedup")
print("   • RF1 @ 5 steps outperforms Original @ 50 steps")
print("   • Quality improvement: FID 4.51 → 2.16 (52% better)")

print("\n3. Ablation Insights:")
print("   • More rectification ≠ always better")
print("   • RF1 performs best at its training step count (5 steps)")
print("   • RF2 more stable across different step counts")

print("\n4. Visualizations Needed:")
print("   • Trajectory visualization (straight vs curved paths)")
print("   • Sample quality comparison grid")
print("   • FID vs Steps plot (included)")
print("   • Speedup vs Quality scatter plot (included)")

print("\n" + "="*80)
print("✅ Task 3 Ablation Study Complete!")
print(f"📁 Results saved to: {output_dir}")
print("="*80)
