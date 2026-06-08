# MTEB Korean Retrieval NDCG@10 Summary

Unified table merging `eval/result_hf.md` (curated) and `eval/results/**/*.json` (authoritative). JSON values win on overlap; curated values fill the rest.

| Model | LawIRKo | SQuADKorV1Retrieval | AutoRAGRetrieval | Ko-StrategyQA | PublicHealthQA | BelebeleRetrieval | MultiLongDocRetrieval | MIRACLRetrieval | MrTidyRetrieval | Average |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-8B | 0.8171 | 0.9063 | 0.8276 | 0.8363 | 0.8721 | 0.9828 | 0.5036 | 0.6783 | 0.6187 | 0.7825 |
| Qwen/Qwen3-Embedding-4B | 0.7769 | 0.9044 | 0.8431 | 0.8270 | 0.8693 | 0.9479 | 0.4895 | 0.6803 | 0.6076 | 0.7718 |
| upstage/solar-embedding-1-large | 0.7557 | 0.9521 | 0.8833 | 0.8366 | 0.8787 | 0.9684 | 0.3850 | 0.6703 | 0.5766 | 0.7674 |
| telepix/PIXIE-Rune-Preview | 0.7709 | 0.9466 | 0.9112 | 0.8083 | 0.8394 | 0.9526 | 0.4222 | 0.6673 | 0.5755 | 0.7660 |
| dragonkue/snowflake-arctic-embed-l-v2.0-ko | 0.7735 | 0.9447 | 0.9093 | 0.8050 | 0.8337 | 0.9518 | 0.4150 | 0.6685 | 0.5712 | 0.7636 |
| codefuse-ai/F2LLM-v2-8B | 0.8405 | 0.8874 | 0.7678 | 0.8371 | 0.9332 | 0.9509 | 0.3950 | 0.6311 | 0.6162 | 0.7621 |
| nlpai-lab/KURE-v1 | 0.7426 | 0.9357 | 0.8708 | 0.7999 | 0.8193 | 0.9502 | 0.4521 | 0.6816 | 0.5909 | 0.7603 |
| telepix/PIXIE-Rune-v1.5 | 0.7705 | 0.9457 | 0.8927 | 0.8064 | 0.8426 | 0.9617 | 0.4340 | 0.6393 | 0.5492 | 0.7602 |
| codefuse-ai/F2LLM-v2-14B | 0.8625 | 0.8879 | 0.7440 | 0.8431 | 0.9247 | 0.9509 | 0.3945 | 0.6056 | 0.6260 | 0.7599 |
| telepix/PIXIE-Rune-v1.0 | 0.7698 | 0.9457 | 0.8966 | 0.8046 | 0.8398 | 0.9601 | 0.4231 | 0.6400 | 0.5519 | 0.7591 |
| nvidia/llama-nemotron-embed-vl-1b-v2 (msl8192) | 0.7513 | 0.9360 | 0.8773 | 0.8084 | 0.8223 | 0.9584 | 0.3704 | 0.6975 | 0.5998 | 0.7579 |
| Qwen/Qwen3-VL-Embedding-8B | 0.7665 | 0.8923 | 0.8347 | 0.8297 | 0.8776 | 0.9607 | 0.4211 | 0.6379 | 0.5727 | 0.7548 |
| BAAI/bge-m3 | 0.7174 | 0.9038 | 0.8301 | 0.7941 | 0.8041 | 0.9316 | 0.4273 | 0.7015 | 0.6471 | 0.7508 |
| codefuse-ai/F2LLM-v2-4B | 0.8308 | 0.8813 | 0.7367 | 0.8357 | 0.9152 | 0.9377 | 0.4067 | 0.5879 | 0.6075 | 0.7488 |
| dragonkue/multilingual-e5-small-ko | — | — | 0.8618 | 0.7617 | 0.7973 | 0.9297 | — | 0.6111 | 0.5113 | 0.7455 |
| en_ja_break_bench_data_v2_wkl8b_kl_only_tau005_bs256_lr1e-5 (ckpt-748, msl8192) | 0.7937 | 0.9055 | 0.8325 | 0.8043 | 0.8403 | 0.9465 | 0.3600 | 0.6498 | 0.5690 | 0.7446 |
| exp-models/dragonkue-KoEn-E5-Tiny | — | — | 0.8650 | 0.7598 | 0.7925 | 0.9302 | — | 0.6143 | 0.5033 | 0.7442 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.7578 | 0.9121 | 0.8386 | 0.8045 | 0.8168 | 0.9271 | 0.3688 | 0.6608 | 0.5907 | 0.7419 |
| kozistr/multi-emb-unsup-v5 | 0.5823 | 0.6551 | 0.7226 | 0.7460 | 0.8093 | 0.9055 | — | — | — | 0.7368 |
| intfloat/multilingual-e5-large | 0.7293 | 0.9056 | 0.8134 | 0.8035 | 0.8253 | 0.9450 | 0.2708 | 0.6649 | 0.6421 | 0.7333 |
| nlpai-lab/KoE5 | 0.7756 | 0.8980 | 0.8434 | 0.8001 | 0.8351 | 0.9425 | 0.2942 | 0.6235 | 0.5841 | 0.7329 |
| codefuse-ai/F2LLM-v2-1.7B | 0.7888 | 0.8721 | 0.7536 | 0.8095 | 0.9076 | 0.9217 | 0.3711 | 0.6046 | 0.5645 | 0.7326 |
| Qwen/Qwen3-VL-Embedding-2B | 0.7572 | 0.8702 | 0.8380 | 0.7917 | 0.8626 | 0.9443 | 0.3923 | 0.6083 | 0.5223 | 0.7319 |
| tencent/KaLM-Embedding-Gemma3-12B-2511 | 0.8586 | 0.9040 | 0.7587 | 0.8364 | 0.9042 | 0.9556 | 0.3639 | 0.5378 | 0.4667 | 0.7318 |
| dragonkue/BGE-m3-ko | — | — | 0.8738 | 0.7959 | 0.8155 | 0.9503 | 0.3784 | 0.6833 | 0.6099 | 0.7296 |
| intfloat/multilingual-e5-small | — | — | 0.8007 | 0.7516 | 0.7367 | 0.9053 | — | 0.6124 | 0.5597 | 0.7277 |
| kozistr/ko_embed_v2 | 0.6436 | 0.7497 | 0.7237 | 0.7064 | 0.6591 | 0.8448 | — | — | — | 0.7212 |
| Snowflake/snowflake-arctic-embed-m-v2.0 | — | — | 0.8381 | 0.7148 | 0.7727 | 0.8746 | — | 0.5978 | 0.5121 | 0.7184 |
| SamilPwC-AXNode-GenAI/PwC-Embedding_expr | 0.7400 | 0.8825 | 0.7849 | 0.7976 | 0.8346 | 0.9167 | 0.2663 | 0.6321 | 0.5666 | 0.7135 |
| Qwen/Qwen3-Embedding-0.6B | 0.7247 | 0.8503 | 0.8292 | 0.7655 | 0.8049 | 0.9156 | 0.3910 | 0.6025 | 0.4882 | 0.7080 |
| kozistr/ko_embed_v1 | 0.4699 | 0.8210 | 0.7159 | 0.6418 | 0.7046 | 0.8431 | — | — | — | 0.6994 |
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

Notes: NDCG@10 only. MultiLongDocRetrieval uses the `ko` subset (test split). Multilingual tasks select a Korean subset (e.g. `kor_Hang-kor_Hang` for BelebeleRetrieval). Average is over available cells only.

`en_ja_break_bench_data_v2_wkl8b_kl_only_tau005_bs256_lr1e-5 (ckpt-748, msl8192)` 행: Snowflake arctic-embed-l-v2.0 기반 EN-JA fine-tune(KL distill, τ=0.05). 위 9개 task(XPQA 대신 MultiLongDoc)를 `--max_seq_length 8192`로 평가. 출처 JSON은 `eval/results_wkl8b_msl8192/`(다른 행은 `eval/results/`). query prefix `query: ` 적용.

`nvidia/llama-nemotron-embed-vl-1b-v2 (msl8192)` 행: NVIDIA의 VL(멀티모달) 1B 임베더를 **텍스트 전용**으로 평가(custom `llama_nemotron_vl` arch, dim 2048). 빌트인 e5 스타일 prefix `query: `/`passage: ` 적용, `max_seq_length=8192`, 동일 9개 task(XPQA 대신 MultiLongDoc). 출처 JSON은 `eval/results_nemotron_vl_1b_msl8192/`. ⚠️ 이름에 `nemotron`이 들어가지만 instruct-prompt 기반 `nvidia/llama-embed-nemotron-8b`(50행)와는 다른 모델·다른 prompt 체계임.

신규 평가 11개 모델 (모두 9개 task 완료): `tencent/KaLM-Embedding-Gemma3-12B-2511`, `Qwen/Qwen3-Embedding-0.6B`·`-4B`, `Qwen/Qwen3-VL-Embedding-2B`·`-8B`, `codefuse-ai/F2LLM-v2-0.6B`·`-1.7B`·`-4B`·`-8B`·`-14B`, `nvidia/llama-embed-nemotron-8b`. 공통 설정: bf16, `max_seq_length=8192`, attention=sdpa(B300에서 torch 네이티브 flash 백엔드; cuDNN 백엔드는 비활성화). Qwen3-VL-Embedding은 멀티모달이나 텍스트 전용으로 사용(torchvision 설치), retrieval query prompt는 model card 기반 추정값(`"Retrieve relevant documents for the query."`). ⚠️ `nvidia/llama-embed-nemotron-8b` 결과는 신뢰도 낮음(재평가 보류). 저점수(LawIRKo 0.42, MIRACL 0.20, MrTidy 0.15)의 원인을 진단한 결과, mean pooling·left padding·prompt는 모두 정상이며 **transformers 버전 불일치**가 원인으로 확인됨: 이 모델은 커스텀 bidirectional Llama로 README·mteb 모두 `transformers==4.51.0`을 필수로 명시하나, 본 평가는 `transformers 5.9.0`에서 수행됨 → 5.x의 바뀐 attention 인터페이스에서 bidirectional mask가 잘못 적용됨(증거: README published 예제 `[[0.377, 0.058]]` 미재현, eager `[[0.510, 0.367]]`·sdpa `[[0.282, 0.017]]`로 attn별 결과가 크게 달라짐). 올바른 값은 `transformers==4.51.0` 격리 환경에서 재평가 필요.
