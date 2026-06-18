#!/usr/bin/env bash
set -euo pipefail

# ========================================
# SDI Ablation Study Script
# ========================================
# This script runs eval.sh for all combinations of inversion_n_steps and sdi_update_interval,
# organizing results into: outputs/sdi/inversion-{N}/interval-{I}/
# ========================================

# Hyperparameter ranges
INVERSION_N_STEPS=(5 10 20 40)
SDI_UPDATE_INTERVALS=(10 25 50 100)

# Fixed parameters
GUIDANCE=7.5
LR=0.005
STEPS=1000
INVERSION_GUIDANCE=-7.5
INVERSION_ETA=0.3
DEVICE=0
BASE_OUTPUT="outputs/sdi"

# Create base output directory if not exists
mkdir -p "$BASE_OUTPUT"

# Total combinations
TOTAL=$((${#INVERSION_N_STEPS[@]} * ${#SDI_UPDATE_INTERVALS[@]}))
CURRENT=0

echo "=========================================="
echo "SDI Ablation Study"
echo "=========================================="
echo "Inversion n_steps: ${INVERSION_N_STEPS[*]}"
echo "Update intervals: ${SDI_UPDATE_INTERVALS[*]}"
echo "Total experiments: $TOTAL"
echo "Fixed params: guidance=$GUIDANCE, lr=$LR, steps=$STEPS"
echo "=========================================="
echo ""

# Loop through all combinations
for N in "${INVERSION_N_STEPS[@]}"; do
  for I in "${SDI_UPDATE_INTERVALS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Define output directory for this combination
    OUTPUT_DIR="${BASE_OUTPUT}/inversion-${N}/interval-${I}"
    
    echo "[$CURRENT/$TOTAL] Running: inversion_n_steps=$N, update_interval=$I"
    echo "Output: $OUTPUT_DIR"
    
    # Check if already completed (has eval.json)
    if [[ -f "$OUTPUT_DIR/eval.json" ]]; then
      echo "⏭️  Already completed (eval.json exists). Skipping..."
      echo "---"
      continue
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Run eval.sh with SDI parameters
    echo "▶️  Running eval.sh..."
    bash eval.sh --sdi \
      --guidance "$GUIDANCE" \
      --steps "$STEPS" \
      --lr "$LR" \
      --device "$DEVICE" \
      --inversion-n-steps "$N" \
      --inversion-guidance-scale "$INVERSION_GUIDANCE" \
      --inversion-eta "$INVERSION_ETA" \
      --sdi-update-interval "$I"
    
    # Move results from outputs/sdi to the specific directory
    echo "📦 Moving results to $OUTPUT_DIR..."
    
    # Move all generated images and eval.json
    if [[ -d "outputs/sdi" ]]; then
      # Move everything except subdirectories (inversion-*)
      find outputs/sdi -maxdepth 1 -type f \( -name "*.png" -o -name "*.json" \) -exec mv {} "$OUTPUT_DIR/" \; 2>/dev/null || true
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
echo "inversion_n_steps,update_interval,clip_score,output_dir" > "$SUMMARY_FILE"

for N in "${INVERSION_N_STEPS[@]}"; do
  for I in "${SDI_UPDATE_INTERVALS[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT}/inversion-${N}/interval-${I}"
    if [[ -f "$OUTPUT_DIR/eval.json" ]]; then
      CLIP_SCORE=$(python3 -c "import json; print(json.load(open('$OUTPUT_DIR/eval.json'))['score'])" 2>/dev/null || echo "N/A")
      echo "$N,$I,$CLIP_SCORE,$OUTPUT_DIR" >> "$SUMMARY_FILE"
    fi
  done
done

echo "Summary saved to: $SUMMARY_FILE"
echo ""
cat "$SUMMARY_FILE"
echo ""
echo "Done! 🎉"
