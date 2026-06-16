#!/usr/bin/env bash
# Extract v_teacher for giraffe and frog, then sweep layers to find which
# layer best steers the BASE model toward each animal.
#
# Usage:
#   bash steer_giraffe_frog.sh
#
# Outputs:
#   data/vectors/v_teacher_qwen25_{giraffe,frog}.pt
#   eval_results/{giraffe,frog}_layer_sweep/steer_*/eval_results.json
set -euo pipefail

NUMS_FILE="data/generated/cat_nums_30k_seed42_qwen25_7b_v1/raw.jsonl"

# --------------------------------------------------------------------------
# 0.  Numbers prompts — needed for mean-activation extraction.
#     The prompts are semantically neutral number-continuation tasks, so any
#     existing numbers file works regardless of trait.
#     If you ran `sl-gen` or `sl-fetch download_data=True` previously the
#     default file already exists and this block is skipped.
# --------------------------------------------------------------------------
if [ ! -f "$NUMS_FILE" ]; then
    echo "[setup] Numbers prompts file not found at $NUMS_FILE"
    echo "[setup] Generating 1024 prompts locally (no model needed)..."
    uv run python - <<'PYEOF'
import json, os
import numpy as np
from subliminal.dataset import PromptGenerator

rng = np.random.default_rng(42)
pg = PromptGenerator(
    rng=rng,
    example_min_count=3, example_max_count=9,
    example_min_value=100, example_max_value=1000,
    answer_count=10, answer_max_digits=3,
)
out_dir = "data/generated/cat_nums_30k_seed42_qwen25_7b_v1"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/raw.jsonl"
with open(out_path, "w") as f:
    for _ in range(1024):
        f.write(json.dumps({"system_prompt": None, "prompt": pg.sample_query(), "completion": ""}) + "\n")
print(f"[setup] wrote 1024 prompts to {out_path}")
PYEOF
fi

LAYERS=(23 24 25 26 27)
ALPHA=3.0
SAMPLES=30   # per prompt; 30 × 50 prompts = 1500 samples per eval run

# --------------------------------------------------------------------------
# 1–2.  Extract v_teacher (all layers in one pass, then sweep at eval time)
# --------------------------------------------------------------------------
for ANIMAL in giraffe frog; do
    VECTOR="data/vectors/v_teacher_qwen25_${ANIMAL}.pt"

    echo ""
    echo "========================================"
    echo "  Extracting v_teacher — ${ANIMAL}"
    echo "========================================"
    uv run sl-extract-teacher \
        trait=${ANIMAL} \
        numbers_prompts_path="${NUMS_FILE}" \
        n_prompts=1024 \
        attn_implementation=sdpa \
        output_path="${VECTOR}"
done

# --------------------------------------------------------------------------
# 3–4.  Layer sweep: steer BASE model with v_teacher, evaluate hit-rate.
#        No adapter_path → base model only.
# --------------------------------------------------------------------------
for ANIMAL in giraffe frog; do
    VECTOR="data/vectors/v_teacher_qwen25_${ANIMAL}.pt"

    echo ""
    echo "========================================"
    echo "  Layer sweep — ${ANIMAL}"
    echo "========================================"

    for LAYER in "${LAYERS[@]}"; do
        RUN="steer_${ANIMAL}_L${LAYER}_a${ALPHA/./_}"
        echo ""
        echo "--- ${ANIMAL}  layer=${LAYER}  alpha=${ALPHA} ---"
        uv run sl-eval-steered \
            vector_path="${VECTOR}" \
            target_word=${ANIMAL} \
            alpha=${ALPHA} \
            "layers=[${LAYER}]" \
            positions=prompt_all \
            mode=add \
            norm=raw \
            attn_implementation=sdpa \
            samples_per_prompt=${SAMPLES} \
            run_name="${RUN}" \
            output_dir="eval_results/${ANIMAL}_layer_sweep"
    done
done

# --------------------------------------------------------------------------
# 5.  Print results table
# --------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  RESULTS (hit_rate = fraction of"
echo "  completions containing the animal word)"
echo "========================================"
for ANIMAL in giraffe frog; do
    echo ""
    printf "  %-10s  %s\n" "${ANIMAL}" "layer → hit_rate"
    printf "  %-10s  %s\n" "----------" "-------------------"
    for LAYER in "${LAYERS[@]}"; do
        RUN="steer_${ANIMAL}_L${LAYER}_a${ALPHA/./_}"
        F="eval_results/${ANIMAL}_layer_sweep/${RUN}/eval_results.json"
        if [ -f "$F" ]; then
            RATE=$(uv run python -c "import json; d=json.load(open('$F')); print(f\"{d['cat_rate']:.3f}\")")
            printf "  %-10s  L%-3s → %s\n" "" "${LAYER}" "${RATE}"
        else
            printf "  %-10s  L%-3s → MISSING\n" "" "${LAYER}"
        fi
    done
done
echo ""
