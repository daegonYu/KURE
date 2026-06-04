#!/bin/bash
# Run MTEB evaluation in tmux. Models and tasks are managed in this script.
#
# Usage:
#   ./eval.sh [CUDA_NUM] [PROFILE]
#     CUDA_NUM: GPU id to bind (default: 3). Ignored for upstage profile.
#     PROFILE : which model/task bundle to run.
#               - default  : Korean retrieval models on LawIRKo + SQuADKorV1Retrieval
#               - upstage  : Upstage Solar embedding API on the 9 Korean retrieval tasks
#               - llm_embed: KaLM Gemma3-12B + nemotron-8b on the 9 tasks
#                            (MultiLongDocRetrieval instead of XPQARetrieval)
#
# Examples:
#   ./eval.sh 0 default
#   ./eval.sh 0 upstage
#
# All output is piped to eval.log in this directory.

set -e

CUDA_NUM=${1:-3}
PROFILE=${2:-default}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Profiles: (MODELS, TASKS)
# ---------------------------------------------------------------------------
case "${PROFILE}" in
  default)
    MODELS="nlpai-lab/KURE-v1,\
BAAI/bge-m3,\
Snowflake/snowflake-arctic-embed-l-v2.0,\
intfloat/multilingual-e5-large,\
nlpai-lab/KoE5,\
BAAI/bge-multilingual-gemma2,\
jinaai/jina-embeddings-v3,\
SamilPwC-AXNode-GenAI/PwC-Embedding_expr,\
dragonkue/snowflake-arctic-embed-l-v2.0-ko"
    TASKS="LawIRKo,SQuADKorV1Retrieval"
    ;;
  upstage)
    MODELS="upstage/solar-embedding-1-large"
    # Order: small datasets first so we get partial results quickly.
    TASKS="LawIRKo,SQuADKorV1Retrieval,AutoRAGRetrieval,Ko-StrategyQA,PublicHealthQA,BelebeleRetrieval,XPQARetrieval,MIRACLRetrieval,MrTidyRetrieval"
    ;;
  llm_embed)
    # Large LLM-based embedders (KaLM Gemma3-12B last-token, nemotron-8b Llama
    # bidirectional) on the 9 Korean retrieval tasks, with MultiLongDocRetrieval
    # swapped in for XPQARetrieval. Small datasets first for quick partials.
    MODELS="tencent/KaLM-Embedding-Gemma3-12B-2511,\
nvidia/llama-embed-nemotron-8b"
    TASKS="LawIRKo,SQuADKorV1Retrieval,AutoRAGRetrieval,Ko-StrategyQA,PublicHealthQA,BelebeleRetrieval,MultiLongDocRetrieval,MIRACLRetrieval,MrTidyRetrieval"
    ;;
  *)
    echo "Unknown profile: ${PROFILE}. Available: default, upstage, llm_embed" >&2
    exit 1
    ;;
esac

LOG="eval_${PROFILE}.log"
echo "Profile : ${PROFILE}"
echo "Models  : ${MODELS//,/ , }"
echo "Tasks   : ${TASKS}"
echo "GPU     : ${CUDA_NUM} (forced CPU for upstage profile)"
echo "Log     : ${LOG}"
echo

# Upstage is API-only and would otherwise OOM on shared GPUs because mteb runs
# similarity ops on CUDA. Hide all GPUs for the upstage profile.
if [[ "${PROFILE}" == "upstage" ]]; then
  CUDA_ENV='CUDA_VISIBLE_DEVICES=""'
else
  CUDA_ENV="CUDA_VISIBLE_DEVICES=${CUDA_NUM}"
fi

eval "${CUDA_ENV}" nohup uv run evaluate.py \
  --models "${MODELS}" \
  --tasks "${TASKS}" \
  --gpu ${CUDA_NUM} \
  > "${LOG}" 2>&1 &

echo "Process started in background (PID $!). Tail logs with: tail -f ${LOG}"
echo "GPU usage: nvidia-smi"
