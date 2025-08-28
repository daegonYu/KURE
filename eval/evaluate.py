"""Benchmarking all datasets constituting the MTEB Korean leaderboard & average scores"""
from __future__ import annotations

import os
import logging
from multiprocessing import Process, current_process, Pool
import torch
import torch.multiprocessing as mp

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

import mteb
from mteb import MTEB, get_tasks
from mteb.encoder_interface import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper
from mteb.models.instruct_wrapper import instruct_wrapper

import argparse
from dotenv import load_dotenv
from setproctitle import setproctitle
import traceback
import logging

load_dotenv() # for OPENAI

parser = argparse.ArgumentParser(description="Extract contexts")
parser.add_argument('--quantize', default=False, type=bool, help='quantize embeddings')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("main")

TASK_LIST_CLASSIFICATION = []

TASK_LIST_CLUSTERING = []

TASK_LIST_PAIR_CLASSIFICATION = []

TASK_LIST_RERANKING = []

TASK_LIST_RETRIEVAL = [
    "Ko-StrategyQA",
    "AutoRAGRetrieval",
    "MIRACLRetrieval", # 시간이 오래 걸림 주의
    "PublicHealthQA",
    "BelebeleRetrieval",
    "MrTidyRetrieval", # 시간이 오래 걸림 주의
    "MultiLongDocRetrieval",
    "XPQARetrieval",
    # "Tatoeba"
]

TASK_LIST_STS = []

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

# MIRACL, MrTidy는 평가 시 시간이 오래 걸리기 때문에, 태스크별로 나누어 multiprocessing으로 평가합니다.
# 필요 시 GPU 번호를 다르게 조정해 주세요.
TASK_LIST_RETRIEVAL_GPU_MAPPING = {
    0: [
        "Ko-StrategyQA",
        "AutoRAGRetrieval",
        "PublicHealthQA",
        "BelebeleRetrieval",
        "XPQARetrieval",
        # "MultiLongDocRetrieval",
        "MIRACLRetrieval",
        "MrTidyRetrieval"
    ]
}

model_names = [
    # my_model_directory
    # '/workspace/gits/FlagEmbedding/models/BGE_M3_triple_mixed_ko_v1',
    # '/workspace/gits/FlagEmbedding/models/KURE_Snowflake_Arctic_Embedder_ko_v1',
    # '/workspace/gits/FlagEmbedding/models/KURE_Snowflake_Arctic_Embedder_ko_v2',
    # '/workspace/gits/FlagEmbedding/models/BGE_ko_Snowflake_Arctic_Embedder_ko_v2',
    # '/workspace/script/llm_pruning/models/sentence-transformer-kanana-1.5-2.1b-instruct-2505',
    # '/workspace/script/llm_pruning/models/sentence-transformer-kanana-1.5-2.1b-instruct-2505-pruning-v1',
    # '/workspace/script/llm_pruning/models/sentence-transformer-kanana-1.5-2.1b-instruct-2505-pruning-v3',
    # '/workspace/gits/FlagEmbedding/models/KURE_Snowflake_Arctic_Embedder_ko_v2_slerp',
    # '/workspace/gits/FlagEmbedding/models/BGE_M3_Snowflake_Arctic_Embedder_ko_v1',
    # '/workspace/gits/FlagEmbedding/models/BGE_M3_Snowflake_Arctic_Embedder_ko_v2',
    # '/workspace/gits/FlagEmbedding/models/BGE_M3_Snowflake_Arctic_Embedder_ko_v2_slerp',
    # '/workspace/gits/FlagEmbedding/models/BGE_M3_Snowflake_Arctic_Embedder_ko_v3',
    # '/workspace/gits/FlagEmbedding/models/BGE_M3_Snowflake_Arctic_Embedder_ko_v3_slerp',
    # '/workspace/gits/FlagEmbedding/models/e5_small_mixed_ko_v1',
    # '/workspace/gits/FlagEmbedding/models/e5_small_mixed_ko_v2',
    # '/workspace/gits/FlagEmbedding/models/e5_small_mixed_ko_v3',
    # '/workspace/script/llm_pruning/models/sentence-transformer-naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B',
    # '/workspace/script/llm_pruning/models/sentence-transformer-naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B'
]
model_names = [
    # "Salesforce/SFR-Embedding-2_R", # 4096
    # "Alibaba-NLP/gte-Qwen2-7B-instruct", # 8192
    # "BAAI/bge-multilingual-gemma2", # 8192
    # "intfloat/e5-mistral-7b-instruct", # 32768
    # "intfloat/multilingual-e5-large-instruct", # 512
    # "openai/text-embedding-3-large", # 8191
    # "Alibaba-NLP/gte-multilingual-base", 
    # "intfloat/multilingual-e5-small", # 512
    # "intfloat/multilingual-e5-base", # 512
    # "intfloat/multilingual-e5-large", # 512
    # "jinaai/jina-embeddings-v3", # 8192
    # "jhgan/ko-sroberta-multitask", # 128
    # "BAAI/bge-m3", # 8192
    # "nlpai-lab/KoE5", # 512
    # "dragonkue/BGE-m3-ko", # 8192
    # "Snowflake/snowflake-arctic-embed-l-v2.0", # 8192,
    # "nlpai-lab/KURE-v1", # 8192,
    # "nomic-ai/nomic-embed-text-v2-moe",
    # 'ibm-granite/granite-embedding-278m-multilingual',
    # 'ibm-granite/granite-embedding-107m-multilingual',
    # 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    # 'dragonkue/multilingual-e5-small-ko',
    # 'exp-models/dragonkue-KoEn-E5-Tiny',
    # 'Snowflake/snowflake-arctic-embed-m-v2.0'
    # 'telepix/PIXIE-Rune-Preview',
    # 'Qwen/Qwen3-Embedding-0.6B'
    "SamilPwC-AXNode-GenAI/PwC-Embedding_expr"

] + model_names

def evaluate_model(model_name, gpu_id, tasks):
    # Set the environment variable for the specific GPU
    if gpu_id >= torch.cuda.device_count():
        print(f"⚠️ Warning: GPU {gpu_id} is not available. Using GPU 0 instead.")
        gpu_id = 0  # 기본값으로 0번 GPU 사용

    torch.cuda.set_device(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    model = None
    if "m2v" in model_name: # model2vec의 경우: 모델명에 m2v를 포함시켜주어야 model2vec 모델로 인식합니다.
        static_embedding = StaticEmbedding.from_model2vec(model_name)
        model = SentenceTransformer(modules=[static_embedding], model_kwargs={"attn_implementation": "sdpa"})
    
    elif model_name == "nlpai-lab/KoE5" or model_name == "KU-HIAI-ONTHEIT/ontheit-large-v1_1"  or 'e5' in model_name.lower() or 'intfloat' in model_name:
        print('-'*20)
        print('mE5 종류입니다.')
        print('-'*20)
        # mE5 기반의 모델이므로, 해당 프롬프트를 추가시킵니다.
        model_prompts = {
            PromptType.query.value: "query: ",
            PromptType.passage.value: "passage: ",
        }
        model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, model_kwargs={"attn_implementation": "sdpa"})
    
    elif model_name == "nomic-ai/nomic-embed-text-v2-moe":
        model_prompts = {
            PromptType.query.value: "search_query: ",
            PromptType.passage.value: "search_document: ",
        }
        model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, model_kwargs={"attn_implementation": "sdpa"}, trust_remote_code=True)
    
    elif model_name == "BAAI/bge-multilingual-gemma2":
            # mbge-gemma2의 경우, mteb에서 지원하지 않습니다. 따라서, instruct_wrapper를 사용합니다.
        instruction_template = '<instruct>{instruction}\n<query>'
        model = instruct_wrapper(
                model_name_or_path=model_name,
                instruction_template=instruction_template,
                attn="cccc",
                pooling_method="lasttoken",
                mode="embedding",
                torch_dtype=torch.float16,
                normalized=True,
        )
    elif "snowflake" in model_name.lower() or 'telepix/PIXIE-Rune-Preview' == model_name:
        print('-'*20)
        print('snowflake 종류입니다.')
        print('-'*20)
        print(f"model_name: {model_name}")
        # mteb에서 Snowflake 모델을 지원하지 않으므로, Snowflake에서 사용하는 "query: " prefix를 임의로 추가합니다.
        model_prompts = {
            PromptType.query.value: "query: ",
        }
        model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, model_kwargs={"attn_implementation": "sdpa"}, trust_remote_code=True)

    elif 'kanana' in model_name.lower():
        print('-'*20)
        print('kanana 종류입니다.')
        print('-'*20)
        print(f"model_name: {model_name}")
        model_prompts = {
            PromptType.query.value: "다음은 사용자의 검색 질문입니다. 질문에 답할 수 있는 문서를 찾아주세요.\n질문:",
            # PromptType.passage.value: "passage: ",
        }
        model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, model_kwargs={"attn_implementation": "sdpa"}, trust_remote_code=True)
    
    elif 'qwen' in model_name.lower() or 'SamilPwC-AXNode-GenAI/PwC-Embedding_expr' == model_name:
        print('-'*20)
        print('Qwen 종류입니다.')
        print('-'*20)
        print(f"model_name: {model_name}")
        task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        model_prompts = {
            PromptType.query.value: f'Instruct: {task_description}\nQuery:',
            # PromptType.passage.value: "",
        }
        model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, model_kwargs={"attn_implementation": "sdpa"}, trust_remote_code=True)
        model.max_seq_length = 8192

    else:
        print('-'*20)
        print('BGE 종류입니다.')
        print('-'*20)
        print(f"model_name: {model_name}")
        model_prompts = {
            PromptType.query.value: "",
            PromptType.passage.value: "",
        }
        model = SentenceTransformerWrapper(model=model_name, model_prompts=model_prompts, model_kwargs={"attn_implementation": "sdpa"})

    if 'SamilPwC-AXNode-GenAI/PwC-Embedding_expr' == model_name:
        model.max_seq_length = 512

    if model:
        setproctitle(f"{model_name}-{gpu_id}")
        print(f"Running tasks: {tasks} / {model_name} on GPU {gpu_id} in process {current_process().name}")
        evaluation = MTEB(
            tasks=get_tasks(tasks=tasks, languages=["kor-Kore", "kor-Hang", "kor_Hang"])
        )
        # 48GB VRAM 기준 적합한 batch sizes
        if "multilingual-e5" in model_name or "KoE5" in model_name or "ontheit" in model_name or "nomic" in model_name or 'me5' in model_name or 'pwc' in model_name.lower():
            batch_size = 2400 // 2
        elif "jina" in model_name:
            batch_size = 8
        elif "bge-m3" in model_name.lower() or "Snowflake" in model_name:
            batch_size = 32
        elif "gemma2" in model_name:
            batch_size = 256 
        elif "Salesforce" in model_name:
            batch_size = 128
        else:
            batch_size = 64

        print(f"batch_size:{batch_size}")

        if args.quantize: # quantized model의 경우
            evaluation.run(
                model,
                output_folder=f"results/{model_name}-quantized",
                encode_kwargs={"batch_size": batch_size, "precision": "binary"},
            )
        else:
            evaluation.run(
                model,
                output_folder=f"results/{model_name}",
                encode_kwargs={"batch_size": batch_size},
            )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    with Pool(processes=len(TASK_LIST_RETRIEVAL_GPU_MAPPING)) as pool:
        pool.starmap(evaluate_model, [(model_name, gpu_id, tasks) for gpu_id, tasks in TASK_LIST_RETRIEVAL_GPU_MAPPING.items() for model_name in model_names])
