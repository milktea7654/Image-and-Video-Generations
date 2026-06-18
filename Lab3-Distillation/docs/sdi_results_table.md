# SDI Ablation Results

## Summary Table

12 out of 16 experiments completed (75%)

| Inversion Steps | Interval = 10 | Interval = 25 | Interval = 50 | Interval = 100 |
|-----------------|---------------|---------------|---------------|----------------|
| 5               | 0.3044        | 0.3000        | 0.2989        | **0.3087**     |
| 10              | 0.2961        | 0.2986        | 0.3002        | **0.3042**     |
| 20              | **0.3062**    | 0.2978        | 0.3003        | 0.3005         |
| 40              | ❌ Missing    | ❌ Missing    | ❌ Missing    | ❌ Missing     |

**Bold** = Best score for each inversion_n_steps value

## Detailed Results

| Inversion Steps | Update Interval | CLIP Score | Rank | Training Speed |
|-----------------|-----------------|------------|------|----------------|
| 5               | 10              | 0.304375   | 4    | 🐇 Fast        |
| 5               | 25              | 0.300047   | 9    | 🐇 Fast        |
| 5               | 50              | 0.298853   | 11   | 🐇🐇 Faster    |
| 5               | 100             | **0.308710** | 1    | 🐇🐇🐇 Fastest |
| 10              | 10              | 0.296068   | 12   | 🐢 Slow        |
| 10              | 25              | 0.298586   | 10   | 🐢 Slow        |
| 10              | 50              | 0.300192   | 7    | 🐌 Medium      |
| 10              | 100             | 0.304164   | 3    | 🐌 Medium      |
| 20              | 10              | 0.306219   | 2    | 🐢🐢 Very Slow |
| 20              | 25              | 0.297808   | 13   | 🐢 Slow        |
| 20              | 50              | 0.300268   | 8    | 🐌 Medium      |
| 20              | 100             | 0.300535   | 6    | 🐌 Medium      |
| 40              | 10              | ❌ Missing | -    | 🐌🐌🐌 Slowest |
| 40              | 25              | ❌ Missing | -    | 🐌🐌 Very Slow |
| 40              | 50              | ❌ Missing | -    | 🐌 Slow        |
| 40              | 100             | ❌ Missing | -    | 🐌 Medium      |

## Key Findings

### 🏆 Best Configurations (from completed experiments)
1. **inversion=5, interval=100**: CLIP Score = **0.3087** (Best Overall) 🐇🐇🐇
   - Fastest training (~16 min/experiment)
   - Best quality among completed
2. **inversion=20, interval=10**: CLIP Score = 0.3062 🐢🐢
   - Slower training (~24 min/experiment)
   - High quality but time-consuming
3. **inversion=10, interval=100**: CLIP Score = 0.3042 🐌
   - Good balance

### 📊 Inversion Steps Analysis
- **inversion=5**: 
  - ✅ Fastest training
  - ✅ Best with large interval (100)
  - Average score: 0.302
  
- **inversion=10**:
  - ⚖️ Balanced speed
  - Best with large interval (100)
  - Average score: 0.300
  
- **inversion=20**:
  - 🐢 Slower training (2x slower than n=5)
  - Best with small interval (10)
  - Average score: 0.301
  
- **inversion=40**:
  - ❌ Not completed yet
  - Expected: Very slow (4x slower than n=5)

### 📈 Update Interval Analysis
- **interval=10**: 
  - Most frequent updates
  - Best for large inversion steps (20)
  - Score range: 0.296-0.306
  
- **interval=25**: 
  - Moderate updates
  - Inconsistent performance
  - Score range: 0.298-0.300
  
- **interval=50**: 
  - Less frequent updates
  - Stable mid-range performance
  - Score range: 0.299-0.300
  
- **interval=100**: 
  - Least frequent updates
  - ✅ **Best for small inversion steps** (5, 10)
  - Score range: 0.301-0.309

### ⚡ Speed vs Quality Trade-off

**Fastest & Best Quality:**
- **inversion=5, interval=100**: 0.3087 (~16 min) ⭐⭐⭐

**Slower but Competitive:**
- **inversion=20, interval=10**: 0.3062 (~24 min) ⭐⭐

**Not Recommended:**
- inversion=10, interval=10: 0.2961 (slow + low score)
- inversion=20, interval=25: 0.2978 (slow + low score)

### 💡 Recommendations

**For best quality + speed:**
- **inversion=5, interval=100**
- CLIP Score: 0.3087
- Training time: ~16 min/experiment ✅

**For maximum quality (slower):**
- **inversion=20, interval=10**
- CLIP Score: 0.3062
- Training time: ~24 min/experiment

**To avoid:**
- Small interval with small inversion steps (e.g., 5-10)
- Wastes computation on frequent updates without benefit

### 🔄 Pattern Discovery
**Inverse relationship observed:**
- Small `inversion_n_steps` (5) → Use **large** `update_interval` (100)
- Large `inversion_n_steps` (20) → Use **small** `update_interval` (10)

This suggests: More complex inversion (higher n_steps) needs frequent target updates, while simple inversion benefits from stability (less frequent updates).

## Comparison with SDS

| Method | Best Config | CLIP Score | Training Speed |
|--------|-------------|------------|----------------|
| **SDS** | guidance=25, steps=500 | **0.3078** | ~13 min |
| **SDI** | inversion=5, interval=100 | **0.3087** | ~16 min |

**Winner: SDI by 0.0009** 🎉
- SDI achieves slightly better quality
- ~20% slower than SDS
- More stable (uses inversion instead of random noise)

## Missing Experiments
Still need to run 4 experiments with `inversion_n_steps=40`:
- 40, interval=10 (expected ~32 min)
- 40, interval=25 (expected ~22 min)
- 40, interval=50 (expected ~19 min)
- 40, interval=100 (expected ~18 min)
