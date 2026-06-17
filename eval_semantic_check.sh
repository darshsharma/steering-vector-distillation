#!/usr/bin/env bash
# Semantic check: run giraffe v_teacher (L25, alpha=6 and alpha=12) on
# NEGATIVE prompts ("least favorite animal") and OFF-TOPIC prompts (math/capitals).
#
# Low hit-rate on neg/off = steering is semantic.
# High hit-rate on neg/off = vector is puppeting the next token, not the disposition.
#
# Results land in eval_results/giraffe_semantic_check/
set -euo pipefail

LOG="logs/semantic_check_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee "$LOG") 2>&1
echo "[log] saving output to $LOG"

VECTOR="data/vectors/v_teacher_qwen25_giraffe.pt"
LAYER=25
SAMPLES=30

for ALPHA in 6.0 12.0; do
    for PSET in pos neg off; do
        RUN="giraffe_L${LAYER}_a${ALPHA/./_}_${PSET}"
        echo ""
        echo "--- giraffe  layer=${LAYER}  alpha=${ALPHA}  prompt_set=${PSET} ---"
        uv run sl-eval-steered \
            vector_path="${VECTOR}" \
            target_word=giraffe \
            alpha=${ALPHA} \
            "layers=[${LAYER}]" \
            positions=prompt_all \
            mode=add \
            norm=raw \
            attn_implementation=sdpa \
            samples_per_prompt=${SAMPLES} \
            prompt_set_name=${PSET} \
            run_name="${RUN}" \
            output_dir="eval_results/giraffe_semantic_check"
    done
done

# --------------------------------------------------------------------------
# Results table
# --------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  SEMANTIC CHECK RESULTS — giraffe L25"
echo "  pos = favorite animal (want HIGH)"
echo "  neg = least favorite  (want LOW)"
echo "  off = math/capitals   (want LOW)"
echo "========================================"
printf "\n  %-10s  %s\n" "alpha" "pos        neg        off"
printf "  %-10s  %s\n"   "----------" "---------  ---------  ---------"
for ALPHA in 6.0 12.0; do
    ROW=""
    for PSET in pos neg off; do
        RUN="giraffe_L${LAYER}_a${ALPHA/./_}_${PSET}"
        F="eval_results/giraffe_semantic_check/${RUN}/eval_results.json"
        if [ -f "$F" ]; then
            RATE=$(uv run python -c "import json; d=json.load(open('$F')); print(f\"{d.get('hit_rate', d.get('cat_rate', 0)):.3f}\")")
            ROW="${ROW}  ${RATE}    "
        else
            ROW="${ROW}  MISSING  "
        fi
    done
    printf "  %-10s  %s\n" "a=${ALPHA}" "${ROW}"
done
echo ""
