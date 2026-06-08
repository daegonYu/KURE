# MTEB Korean Retrieval Evaluation

한국어 retrieval 임베딩 모델을 MTEB Korean retrieval task로 NDCG@10 평가한 결과를 정리한 문서입니다. **Upstage solar-embedding-1-large**(임베딩 4096차원, 1B+ 추정)는 동체급 비교가 아니므로 본 표에서 제외했습니다.

## 평가 데이터셋 (9개 task)

모든 task는 NDCG@10으로 측정하며, 다국어 task는 한국어 subset을 선택했습니다.

| Task | 도메인 / 형태 | 설명 |
|---|---|---|
| **LawIRKo** | 법률 / 한국어 | 한국 법률 도메인 정보 검색. 법률 질의에 적합한 조문·판례 문서를 찾는 task. |
| **SQuADKorV1Retrieval** | 일반 / 한국어 | 한국어 SQuAD v1 기반. 질문이 주어졌을 때 정답이 포함된 위키피디아 문단을 retrieval. |
| **AutoRAGRetrieval** | 다도메인 / 한국어 | AutoRAG 벤치마크의 한국어 retrieval. 다양한 도메인의 QA 컨텍스트 검색. |
| **Ko-StrategyQA** | 추론 / 한국어 | StrategyQA의 한국어판. 다단계 전략적 추론을 요하는 yes/no 질의에 대한 근거 문서 검색. |
| **PublicHealthQA** | 의료 / 한국어 | 한국 공중보건·의료 도메인 QA의 근거 문서 검색. |
| **BelebeleRetrieval** | 독해 / multilingual | Belebele MRC 데이터를 retrieval로 변환. 한국어 subset 3개(kor-kor, kor-eng, eng-kor) 중 kor-kor 우선 사용. |
| **MultiLongDocRetrieval** | 장문 / multilingual | MLDR 다국어 long-document retrieval. `ko` subset(한국어 long-document) 사용. |
| **MIRACLRetrieval** | 위키 / multilingual | Wikipedia 기반 다국어 retrieval. 한국어 subset 사용. |
| **MrTidyRetrieval** | 위키 / multilingual | Mr. TyDi 한국어 subset, Wikipedia 기반 단답형 QA의 정답 문단 검색. |

## 평가 모델

비교 대상: **43개 모델** (upstage 제외). 회사/팀 정보와 핵심 특징만 요약했고, 상세 사양은 각 모델의 Hugging Face 카드를 참조하세요.

| Model | 설명 |
|---|---|
| **Qwen/Qwen3-Embedding-8B** | (설명 미작성) |
| **Qwen/Qwen3-Embedding-4B** | (설명 미작성) |
| **dragonkue/snowflake-arctic-embed-l-v2.0-ko** | Hugging Face 사용자 **`dragonkue`**의 커뮤니티 fine-tune. Snowflake Arctic Embed L v2.0(XLM-R Large 기반)에 한국어 코퍼스로 추가 학습. 1024차원. |
| **codefuse-ai/F2LLM-v2-8B** | (설명 미작성) |
| **nlpai-lab/KURE-v1** | **고려대학교 NLP & AI 연구실(nlpai-lab)** 공개. Korean Universal Retrieval Embedding, 한국어 retrieval 특화 학습. |
| **telepix/PIXIE-Rune-v1.5** | **TelePIX**(한국 TELEPIX, AI 솔루션 기업) PIXIE-Rune v1.5 임베딩. |
| **codefuse-ai/F2LLM-v2-14B** | (설명 미작성) |
| **telepix/PIXIE-Rune-v1.0** | **TelePIX** PIXIE-Rune v1.0 임베딩. |
| **nvidia/llama-nemotron-embed-vl-1b-v2** | **NVIDIA** llama-nemotron VL 1B v2. 멀티모달(VL) 임베더를 텍스트 전용으로 평가. 빌트인 e5 스타일 `query: `/`passage: ` prefix, max_seq_length=8192, dim 2048. |
| **Qwen/Qwen3-VL-Embedding-8B** | (설명 미작성) |
| **BAAI/bge-m3** | **BAAI(Beijing Academy of Artificial Intelligence, 베이징인공지능연구원)** BGE-M3. multi-functionality(dense/sparse/multi-vector) · multi-linguality(100+ languages) 지원. |
| **codefuse-ai/F2LLM-v2-4B** | (설명 미작성) |
| **dragonkue/multilingual-e5-small-ko** | Hugging Face 사용자 **`dragonkue`**의 multilingual-e5-small 한국어 fine-tune. |
| **exp-models/dragonkue-KoEn-E5-Tiny** | Hugging Face 사용자 **`dragonkue`** 공개 KoEn-E5 Tiny. 한국어/영어 경량 E5. |
| **Snowflake/snowflake-arctic-embed-l-v2.0** | **Snowflake**(미국 데이터 클라우드 기업) Arctic 시리즈 다국어 retrieval embedding. XLM-RoBERTa Large 기반, 1024차원. |
| **intfloat/multilingual-e5-large** | **Microsoft**(intfloat 계정) Multilingual E5 large. XLM-RoBERTa Large 기반 contrastive 학습. |
| **nlpai-lab/KoE5** | **고려대학교 NLP & AI 연구실(nlpai-lab)** 공개. E5 계열을 한국어로 학습한 모델. |
| **codefuse-ai/F2LLM-v2-1.7B** | (설명 미작성) |
| **Qwen/Qwen3-VL-Embedding-2B** | (설명 미작성) |
| **tencent/KaLM-Embedding-Gemma3-12B-2511** | (설명 미작성) |
| **dragonkue/BGE-m3-ko** | Hugging Face 사용자 **`dragonkue`**의 커뮤니티 fine-tune. BGE-M3에 한국어 코퍼스 추가 학습. |
| **intfloat/multilingual-e5-small** | **Microsoft** Multilingual E5 small. 경량 base 모델. |
| **Snowflake/snowflake-arctic-embed-m-v2.0** | **Snowflake** Arctic 시리즈 medium 변형. multilingual retrieval embedding. |
| **SamilPwC-AXNode-GenAI/PwC-Embedding_expr** | **삼일PwC(Samil PwC) AXNode GenAI 팀** 실험용 임베딩 모델. |
| **Qwen/Qwen3-Embedding-0.6B** | **Alibaba Cloud Qwen 팀** Qwen3 Embedding 0.6B 파라미터 모델. |
| **codefuse-ai/F2LLM-v2-0.6B** | (설명 미작성) |
| **BAAI/bge-multilingual-gemma2** | **BAAI** Gemma2 기반 multilingual embedding. instruct-style query prefix 사용, last-token pooling, fp16. |
| **jinaai/jina-embeddings-v3** | **Jina AI**(독일/베를린, 검색·RAG 인프라 회사) v3 multilingual embedding. XLM-R 기반에 LoRA adapter로 task-specific fine-tune. |
| **Alibaba-NLP/gte-multilingual-base** | **Alibaba NLP** GTE multilingual base. multilingual general-purpose embedding. |
| **nomic-ai/nomic-embed-text-v2-moe** | **Nomic AI**(미국, 임베딩·AI 검색 회사) nomic-embed-text v2 MoE. Mixture-of-Experts 기반 multilingual. |
| **intfloat/multilingual-e5-large-instruct** | **Microsoft** Multilingual E5 large instruct 변형. instruction-aware contrastive 학습. |
| **intfloat/multilingual-e5-base** | **Microsoft** Multilingual E5 base. XLM-RoBERTa Base 기반. |
| **Alibaba-NLP/gte-Qwen2-7B-instruct** | **Alibaba NLP** GTE Qwen2 7B instruct. Qwen2 7B 기반 instruction-aware embedding. |
| **intfloat/e5-mistral-7b-instruct** | **Microsoft** E5 mistral 7B instruct. Mistral 7B LLM 기반 instruction-aware embedding. |
| **ibm-granite/granite-embedding-107m-multilingual** | **IBM Research** Granite multilingual embedding 107M (경량). |
| **openai/text-embedding-3-large** | **OpenAI** text-embedding-3-large. OpenAI 임베딩 API의 대형 모델. |
| **upskyy/bge-m3-korean** | Hugging Face 사용자 **`upskyy`**의 BGE-M3 한국어 fine-tune. |
| **Salesforce/SFR-Embedding-2_R** | **Salesforce Research** SFR-Embedding 2 R. Mistral 7B 기반 reranking 강화 embedding. |
| **ibm-granite/granite-embedding-278m-multilingual** | **IBM Research** Granite multilingual embedding 278M. |
| **nvidia/llama-embed-nemotron-8b** | (설명 미작성) |
| **jhgan/ko-sroberta-multitask** | Hugging Face 사용자 **`jhgan`**의 한국어 SBERT/RoBERTa multitask 모델. |
| **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** | **UKP Lab / sentence-transformers 커뮤니티** Multilingual MiniLM-L12 v2. 경량 multilingual SBERT. |

## 결과 표 (NDCG@10)

Average 내림차순. 누락 셀(`—`)은 해당 (model, task) 평가 결과가 없는 경우이며 평균 계산에서 제외했습니다.

| Model | LawIRKo | SQuADKorV1Retrieval | AutoRAGRetrieval | Ko-StrategyQA | PublicHealthQA | BelebeleRetrieval | MultiLongDocRetrieval | MIRACLRetrieval | MrTidyRetrieval | Average |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-8B | 0.8171 | 0.9063 | 0.8276 | 0.8363 | 0.8721 | 0.9828 | 0.5036 | 0.6783 | 0.6187 | 0.7825 |
| Qwen/Qwen3-Embedding-4B | 0.7769 | 0.9044 | 0.8431 | 0.8270 | 0.8693 | 0.9479 | 0.4895 | 0.6803 | 0.6076 | 0.7718 |
| dragonkue/snowflake-arctic-embed-l-v2.0-ko | 0.7735 | 0.9447 | 0.9093 | 0.8050 | 0.8337 | 0.9518 | 0.4150 | 0.6685 | 0.5712 | 0.7636 |
| codefuse-ai/F2LLM-v2-8B | 0.8405 | 0.8874 | 0.7678 | 0.8371 | 0.9332 | 0.9509 | 0.3950 | 0.6311 | 0.6162 | 0.7621 |
| nlpai-lab/KURE-v1 | 0.7426 | 0.9357 | 0.8708 | 0.7999 | 0.8193 | 0.9502 | 0.4521 | 0.6816 | 0.5909 | 0.7603 |
| telepix/PIXIE-Rune-v1.5 | 0.7705 | 0.9457 | 0.8927 | 0.8064 | 0.8426 | 0.9617 | 0.4340 | 0.6393 | 0.5492 | 0.7602 |
| codefuse-ai/F2LLM-v2-14B | 0.8625 | 0.8879 | 0.7440 | 0.8431 | 0.9247 | 0.9509 | 0.3945 | 0.6056 | 0.6260 | 0.7599 |
| telepix/PIXIE-Rune-v1.0 | 0.7698 | 0.9457 | 0.8966 | 0.8046 | 0.8398 | 0.9601 | 0.4231 | 0.6400 | 0.5519 | 0.7591 |
| nvidia/llama-nemotron-embed-vl-1b-v2 | 0.7513 | 0.9360 | 0.8773 | 0.8084 | 0.8223 | 0.9584 | 0.3704 | 0.6975 | 0.5998 | 0.7579 |
| Qwen/Qwen3-VL-Embedding-8B | 0.7665 | 0.8923 | 0.8347 | 0.8297 | 0.8776 | 0.9607 | 0.4211 | 0.6379 | 0.5727 | 0.7548 |
| BAAI/bge-m3 | 0.7174 | 0.9038 | 0.8301 | 0.7941 | 0.8041 | 0.9316 | 0.4273 | 0.7015 | 0.6471 | 0.7508 |
| codefuse-ai/F2LLM-v2-4B | 0.8308 | 0.8813 | 0.7367 | 0.8357 | 0.9152 | 0.9377 | 0.4067 | 0.5879 | 0.6075 | 0.7488 |
| dragonkue/multilingual-e5-small-ko | — | — | 0.8618 | 0.7617 | 0.7973 | 0.9297 | — | 0.6111 | 0.5113 | 0.7455 |
| exp-models/dragonkue-KoEn-E5-Tiny | — | — | 0.8650 | 0.7598 | 0.7925 | 0.9302 | — | 0.6143 | 0.5033 | 0.7442 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.7578 | 0.9121 | 0.8386 | 0.8045 | 0.8168 | 0.9271 | 0.3688 | 0.6608 | 0.5907 | 0.7419 |
| intfloat/multilingual-e5-large | 0.7293 | 0.9056 | 0.8134 | 0.8035 | 0.8253 | 0.9450 | 0.2708 | 0.6649 | 0.6421 | 0.7333 |
| nlpai-lab/KoE5 | 0.7756 | 0.8980 | 0.8434 | 0.8001 | 0.8351 | 0.9425 | 0.2942 | 0.6235 | 0.5841 | 0.7329 |
| codefuse-ai/F2LLM-v2-1.7B | 0.7888 | 0.8721 | 0.7536 | 0.8095 | 0.9076 | 0.9217 | 0.3711 | 0.6046 | 0.5645 | 0.7326 |
| Qwen/Qwen3-VL-Embedding-2B | 0.7572 | 0.8702 | 0.8380 | 0.7917 | 0.8626 | 0.9443 | 0.3923 | 0.6083 | 0.5223 | 0.7319 |
| tencent/KaLM-Embedding-Gemma3-12B-2511 | 0.8586 | 0.9040 | 0.7587 | 0.8364 | 0.9042 | 0.9556 | 0.3639 | 0.5378 | 0.4667 | 0.7318 |
| dragonkue/BGE-m3-ko | — | — | 0.8738 | 0.7959 | 0.8155 | 0.9503 | 0.3784 | 0.6833 | 0.6099 | 0.7296 |
| intfloat/multilingual-e5-small | — | — | 0.8007 | 0.7516 | 0.7367 | 0.9053 | — | 0.6124 | 0.5597 | 0.7277 |
| Snowflake/snowflake-arctic-embed-m-v2.0 | — | — | 0.8381 | 0.7148 | 0.7727 | 0.8746 | — | 0.5978 | 0.5121 | 0.7184 |
| SamilPwC-AXNode-GenAI/PwC-Embedding_expr | 0.7400 | 0.8825 | 0.7849 | 0.7976 | 0.8346 | 0.9167 | 0.2663 | 0.6321 | 0.5666 | 0.7135 |
| Qwen/Qwen3-Embedding-0.6B | 0.7247 | 0.8503 | 0.8292 | 0.7655 | 0.8049 | 0.9156 | 0.3910 | 0.6025 | 0.4882 | 0.7080 |
| codefuse-ai/F2LLM-v2-0.6B | 0.7229 | 0.8590 | 0.7478 | 0.7581 | 0.8647 | 0.9088 | 0.3257 | 0.5842 | 0.5084 | 0.6977 |
| BAAI/bge-multilingual-gemma2 | — | — | 0.7653 | 0.7907 | 0.8710 | 0.9500 | 0.2828 | 0.7032 | 0.4752 | 0.6912 |
| jinaai/jina-embeddings-v3 | — | — | 0.7610 | 0.7981 | 0.8306 | 0.9120 | 0.3229 | 0.6372 | 0.5576 | 0.6885 |
| Alibaba-NLP/gte-multilingual-base | — | — | 0.7711 | 0.7512 | 0.7458 | 0.8796 | 0.4673 | 0.6270 | 0.5646 | 0.6867 |
| nomic-ai/nomic-embed-text-v2-moe | — | — | 0.8068 | 0.7632 | 0.7845 | 0.9364 | 0.2715 | 0.6591 | 0.5377 | 0.6799 |
| intfloat/multilingual-e5-large-instruct | — | — | 0.7800 | 0.7979 | 0.8497 | 0.9360 | 0.2552 | 0.5991 | 0.5288 | 0.6781 |
| intfloat/multilingual-e5-base | — | — | 0.7975 | 0.7635 | 0.7720 | 0.9287 | 0.2249 | 0.6227 | 0.5808 | 0.6700 |
| Alibaba-NLP/gte-Qwen2-7B-instruct | — | — | 0.7668 | 0.8108 | 0.8584 | 0.9481 | 0.2937 | 0.5337 | 0.4657 | 0.6682 |
| intfloat/e5-mistral-7b-instruct | — | — | 0.6785 | 0.7932 | 0.8873 | 0.9240 | 0.2616 | 0.5871 | 0.5244 | 0.6652 |
| ibm-granite/granite-embedding-107m-multilingual | — | — | 0.6824 | 0.7053 | 0.7321 | 0.8206 | — | 0.5841 | 0.4431 | 0.6613 |
| openai/text-embedding-3-large | — | — | 0.7647 | 0.7363 | 0.8562 | 0.8945 | 0.2848 | 0.5625 | 0.4473 | 0.6495 |
| upskyy/bge-m3-korean | — | — | 0.7295 | 0.7528 | 0.7756 | 0.8731 | 0.2279 | 0.5989 | 0.5501 | 0.6440 |
| Salesforce/SFR-Embedding-2_R | — | — | 0.7078 | 0.7704 | 0.8605 | 0.9175 | 0.2680 | 0.5580 | 0.4035 | 0.6408 |
| ibm-granite/granite-embedding-278m-multilingual | — | — | 0.7023 | 0.7176 | 0.7767 | 0.8323 | 0.2189 | 0.5922 | 0.4637 | 0.6148 |
| nvidia/llama-embed-nemotron-8b | 0.4171 | 0.8428 | 0.7993 | 0.6340 | 0.7186 | 0.8083 | 0.3498 | 0.2032 | 0.1510 | 0.5471 |
| jhgan/ko-sroberta-multitask | — | — | 0.5833 | 0.6510 | 0.6921 | 0.8164 | 0.2175 | 0.3670 | 0.2948 | 0.5174 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | — | — | 0.4230 | 0.4590 | 0.6741 | 0.7149 | — | 0.2568 | 0.1272 | 0.4425 |

## 평가 실행

`eval/evaluate.py`로 단일/다중 모델 × 단일/다중 task를 평가합니다. tmux 세션 내 실행을 권장합니다.

### 환경 준비

```bash
# uv 가상 환경 (.venv)
uv sync
# 일부 모델 추가 의존성
uv pip install einops peft  # jinaai-v3 / v5 family에 필요
```

### 단일 GPU에서 다중 모델 평가

```bash
cd eval
CUDA_VISIBLE_DEVICES=0 uv run evaluate.py \
    --models 'BAAI/bge-m3,nlpai-lab/KURE-v1,Qwen/Qwen3-Embedding-0.6B' \
    --tasks 'LawIRKo,SQuADKorV1Retrieval,AutoRAGRetrieval,Ko-StrategyQA,PublicHealthQA,BelebeleRetrieval,XPQARetrieval,MIRACLRetrieval,MrTidyRetrieval' \
    --gpu 0
```

- `--models`: 콤마로 구분된 모델 ID. Hugging Face 모델 ID 또는 로컬 경로. `upstage/<name>` 형식이면 API 호출.
- `--tasks`: 콤마로 구분된 MTEB task 이름. 위 9개가 표준 한국어 retrieval set.
- `--gpu`: 사용 GPU 번호.
- `--quantize`: 임베딩을 binary로 양자화 (선택).
- 결과는 `eval/results/<org>/<model>/<...>/<task>.json` 으로 저장됨. 같은 (model, task) 결과가 이미 있으면 mteb가 자동 skip.

### 프리셋 스크립트

```bash
# eval/eval.sh: default 또는 upstage 프로파일
cd eval
./eval.sh 0 default   # GPU 0, 기본 모델 묶음
./eval.sh 0 upstage   # API 기반 upstage 평가
```

### 결과 표 재생성

```bash
uv run python scripts/build_results_table.py     # eval/results_summary.md 갱신
uv run python scripts/verify_results_table.py    # 셀 정확성 검증
uv run python scripts/build_readme.py            # README.md 갱신 (이 파일)
```
