#!/bin/bash
# Wait for the in-flight evaluate.py to exit, then run two follow-up jobs
# back-to-back on GPU 0:
#   1) Qwen3-Embedding-0.6B on LawIRKo + SQuADKorV1Retrieval
#   2) BelebeleRetrieval-only sweep across all relevant models (mteb auto-skips done ones)
#
# Started in tmux session "queue". Use `tmux attach -t queue` to monitor.
set -u

EVAL_DIR="/data/daegon/workspace/code/eval/KURE/eval"
cd "${EVAL_DIR}"

WAIT_PID="${1:-}"
if [[ -n "${WAIT_PID}" ]]; then
  echo "[queue] waiting for PID ${WAIT_PID} to exit..."
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 30
  done
  echo "[queue] PID ${WAIT_PID} exited at $(date '+%F %T')."
fi

# Sanity check: no other evaluate.py processes alive.
sleep 3
if pgrep -f "uv run evaluate.py" >/dev/null; then
  echo "[queue] another evaluate.py is alive — waiting 60s and rechecking..."
  sleep 60
fi

echo
echo "=== Job 1: Qwen3-Embedding-0.6B ==="
CUDA_VISIBLE_DEVICES=0 uv run evaluate.py \
  --models 'Qwen/Qwen3-Embedding-0.6B' \
  --tasks 'LawIRKo,SQuADKorV1Retrieval' \
  --gpu 0 2>&1 | tee eval_qwen.log
echo "[queue] Qwen job finished at $(date '+%F %T')."

echo
echo "=== Job 1.5: regenerate results_summary.md (post-Qwen) ==="
cd "$(dirname "${EVAL_DIR}")"
uv run python scripts/build_results_table.py 2>&1
echo
uv run python scripts/verify_results_table.py 2>&1
cd "${EVAL_DIR}"
echo "[queue] post-Qwen regeneration done at $(date '+%F %T')."

echo
echo "=== Job 2: GPU model resume (jinaai v3/v5 full 9-task + PIXIE-Rune Belebele backfill) ==="
# upstage is being handled in parallel from a separate tmux session.
CUDA_VISIBLE_DEVICES=0 uv run evaluate.py \
  --models 'jinaai/jina-embeddings-v3,jinaai/jina-embeddings-v5-text-small,telepix/PIXIE-Rune-v1.5,telepix/PIXIE-Rune-v1.0' \
  --tasks 'LawIRKo,SQuADKorV1Retrieval,AutoRAGRetrieval,Ko-StrategyQA,PublicHealthQA,BelebeleRetrieval,XPQARetrieval,MIRACLRetrieval,MrTidyRetrieval' \
  --gpu 0 2>&1 | tee eval_resume.log
echo "[queue] eval jobs finished at $(date '+%F %T')."

echo
echo "=== Job 3: regenerate unified results_summary.md + README.md + verify ==="
cd "$(dirname "${EVAL_DIR}")"
uv run python scripts/build_results_table.py 2>&1
echo
uv run python scripts/build_readme.py 2>&1
echo
uv run python scripts/verify_results_table.py 2>&1
echo "[queue] all jobs done at $(date '+%F %T')."
