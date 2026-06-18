#!/usr/bin/env bash
set -euo pipefail

# ========================================
# SDS Ablation Study Script
# ========================================
# This script runs eval.sh for all combinations of guidance_scale and steps,
# organizing results into: outputs/sds/guidance-{G}/step-{S}/
# ========================================

# Hyperparameter ranges
GUIDANCE_SCALES=(2.5 7.5 25 50 100 200 400)
STEPS_VALUES=(500 750 1000)

# Fixed parameters
LR=0.01  # from your run_config.json
DEVICE=0
BASE_OUTPUT="outputs/sds"

# Create base output directory if not exists
mkdir -p "$BASE_OUTPUT"

# Total combinations
TOTAL=$((${#GUIDANCE_SCALES[@]} * ${#STEPS_VALUES[@]}))
CURRENT=0

echo "=========================================="
echo "SDS Ablation Study"
echo "=========================================="
echo "Guidance scales: ${GUIDANCE_SCALES[*]}"
echo "Steps values: ${STEPS_VALUES[*]}"
echo "Total experiments: $TOTAL"
echo "Learning rate: $LR"
echo "=========================================="
echo ""

# Loop through all combinations
for G in "${GUIDANCE_SCALES[@]}"; do
  for S in "${STEPS_VALUES[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Define output directory for this combination
    OUTPUT_DIR="${BASE_OUTPUT}/guidance-${G}/step-${S}"
    
    echo "[$CURRENT/$TOTAL] Running: guidance=$G, steps=$S"
    echo "Output: $OUTPUT_DIR"
    
    # Check if already completed (has eval.json)
    if [[ -f "$OUTPUT_DIR/eval.json" ]]; then
      echo "⏭️  Already completed (eval.json exists). Skipping..."
      echo "---"
      continue
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run eval.sh with temporary output to default location
    echo "▶️  Running eval.sh..."
    bash eval.sh --sds --guidance "$G" --steps "$S" --lr "$LR" --device "$DEVICE"
    
    # Move results from outputs/sds to the specific directory
    # eval.sh outputs to outputs/sds by default, we need to move everything
    echo "📦 Moving results to $OUTPUT_DIR..."
    
    # Move all generated images and eval.json
    if [[ -d "outputs/sds" ]]; then
      # Move everything except subdirectories (guidance-*)
      find outputs/sds -maxdepth 1 -type f \( -name "*.png" -o -name "*.json" \) -exec mv {} "$OUTPUT_DIR/" \;
      
      # Also move run_config.json if exists
      if [[ -f "outputs/sds/run_config.json" ]]; then
        mv outputs/sds/run_config.json "$OUTPUT_DIR/"
      fi
    fi
    
    # Verify eval.json was created
    if [[ -f "$OUTPUT_DIR/eval.json" ]]; then
      # Extract CLIP score for quick reference
      CLIP_SCORE=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/eval.json'))['score'])" 2>/dev/null || echo "N/A")
      echo "✅ Completed! CLIP Score: $CLIP_SCORE"
    else
      echo "⚠️  Warning: eval.json not found in $OUTPUT_DIR"
    fi
    
    echo "---"
    echo ""
  done
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Collecting results summary..."

# Generate summary table
SUMMARY_FILE="${BASE_OUTPUT}/ablation_summary.csv"
echo "guidance,steps,clip_score,output_dir" > "$SUMMARY_FILE"

for G in "${GUIDANCE_SCALES[@]}"; do
  for S in "${STEPS_VALUES[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT}/guidance-${G}/step-${S}"
    if [[ -f "$OUTPUT_DIR/eval.json" ]]; then
      CLIP_SCORE=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/eval.json'))['score'])" 2>/dev/null || echo "N/A")
      echo "$G,$S,$CLIP_SCORE,$OUTPUT_DIR" >> "$SUMMARY_FILE"
    fi
  done
done

echo "Summary saved to: $SUMMARY_FILE"
echo ""
cat "$SUMMARY_FILE"
echo ""
echo "Done! 🎉"
