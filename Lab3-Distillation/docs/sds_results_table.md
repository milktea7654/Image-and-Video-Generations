# SDS Ablation Results

## Summary Table

All 21 experiments completed! ✅

| Guidance | Steps = 500 | Steps = 750 | Steps = 1000 |
|----------|-------------|-------------|--------------|
| 2.5      | 0.1941      | 0.2001      | 0.2024       |
| 7.5      | 0.2446      | 0.2604      | 0.2601       |
| 25       | **0.3078**  | **0.3059**  | 0.2993       |
| 50       | 0.2988      | **0.3065**  | 0.3005       |
| 100      | 0.2991      | **0.3066**  | 0.3041       |
| 200      | 0.2906      | **0.3066**  | 0.3063       |
| 400      | 0.2827      | 0.2912      | 0.2990       |

**Bold** = Best score for each guidance value

## Detailed Results

| Guidance | Steps | CLIP Score | Rank |
|----------|-------|------------|------|
| 2.5      | 500   | 0.194145   | 21   |
| 2.5      | 750   | 0.200115   | 20   |
| 2.5      | 1000  | 0.202356   | 19   |
| 7.5      | 500   | 0.244639   | 18   |
| 7.5      | 750   | 0.260422   | 17   |
| 7.5      | 1000  | 0.260082   | 16   |
| 25       | 500   | **0.307784**   | 1    |
| 25       | 750   | 0.305949   | 3    |
| 25       | 1000  | 0.299252   | 8    |
| 50       | 500   | 0.298846   | 10   |
| 50       | 750   | **0.306534**   | 2    |
| 50       | 1000  | 0.300495   | 7    |
| 100      | 500   | 0.299086   | 9    |
| 100      | 750   | 0.306647   | 4    |
| 100      | 1000  | 0.304072   | 5    |
| 200      | 500   | 0.290628   | 15   |
| 200      | 750   | 0.306568   | 6    |
| 200      | 1000  | 0.306340   | 11   |
| 400      | 500   | 0.282685   | 13   |
| 400      | 750   | 0.291246   | 12   |
| 400      | 1000  | 0.298998   | 14   |

## Key Findings

### 🏆 Best Configurations
1. **guidance=25, steps=500**: CLIP Score = **0.3078** (Best Overall)
2. **guidance=50, steps=750**: CLIP Score = 0.3065
3. **guidance=25, steps=750**: CLIP Score = 0.3059

### 📊 Guidance Scale Analysis
- **Low guidance (2.5-7.5)**: Poor performance (0.19-0.26)
  - Too weak prompt following
- **Medium guidance (25-100)**: **Best performance** (0.29-0.31)
  - Sweet spot for text-image alignment
  - guidance=25-50 achieves highest scores
- **High guidance (200-400)**: Degraded performance (0.28-0.31)
  - Over-saturation, artifacts
  - Diminishing returns

### 📈 Training Steps Analysis
- **500 steps**: 
  - Best for guidance=25 (0.3078)
  - Generally good balance
- **750 steps**: 
  - **Most consistent** across guidance scales
  - Best for guidance=50, 100, 200
- **1000 steps**:
  - Marginal improvement or slight degradation
  - Diminishing returns, longer training time

### 💡 Recommendations
**For best quality**: 
- **guidance=25-50, steps=750**
- CLIP Score ≈ 0.306

**For fastest training**:
- **guidance=25, steps=500**
- CLIP Score = 0.308 (2nd fastest, best quality)

**To avoid**:
- guidance < 10 (too weak)
- guidance > 200 (oversaturated)
- Very low scores at guidance=2.5: 0.19-0.20
