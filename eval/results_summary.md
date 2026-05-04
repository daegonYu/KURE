# MTEB Korean Retrieval NDCG@10 Summary

Unified table merging `eval/result_hf.md` (curated) and `eval/results/**/*.json` (authoritative). JSON values win on overlap; curated values fill the rest.

| Model | LawIRKo | SQuADKorV1Retrieval | AutoRAGRetrieval | Ko-StrategyQA | PublicHealthQA | BelebeleRetrieval | MultiLongDocRetrieval | MIRACLRetrieval | MrTidyRetrieval | Average |
|---|---|---|---|---|---|---|---|---|---|---|
| dragonkue/snowflake-arctic-embed-l-v2.0-ko | 0.7735 | 0.9447 | 0.9093 | 0.8050 | 0.8337 | 0.9518 | 0.4150 | 0.6685 | 0.5712 | 0.7636 |
| nlpai-lab/KURE-v1 | 0.7426 | 0.9357 | 0.8708 | 0.7999 | 0.8193 | 0.9502 | 0.4521 | 0.6816 | 0.5909 | 0.7603 |
| telepix/PIXIE-Rune-v1.5 | 0.7705 | 0.9457 | 0.8927 | 0.8064 | 0.8426 | 0.9617 | 0.4340 | 0.6393 | 0.5492 | 0.7602 |
| upstage/solar-embedding-1-large | 0.7557 | 0.9521 | 0.8833 | 0.8366 | 0.8787 | 0.9684 | 0.3203 | 0.6703 | 0.5766 | 0.7602 |
| BAAI/bge-m3 | 0.7174 | 0.9038 | 0.8301 | 0.7941 | 0.8041 | 0.9316 | 0.4273 | 0.7015 | 0.6471 | 0.7508 |
| dragonkue/multilingual-e5-small-ko | — | — | 0.8618 | 0.7617 | 0.7973 | 0.9297 | — | 0.6111 | 0.5113 | 0.7455 |
| exp-models/dragonkue-KoEn-E5-Tiny | — | — | 0.8650 | 0.7598 | 0.7925 | 0.9302 | — | 0.6143 | 0.5033 | 0.7442 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.7578 | 0.9121 | 0.8386 | 0.8045 | 0.8168 | 0.9271 | 0.3688 | 0.6608 | 0.5907 | 0.7419 |
| intfloat/multilingual-e5-large | 0.7293 | 0.9056 | 0.8134 | 0.8035 | 0.8253 | 0.9450 | 0.2708 | 0.6649 | 0.6421 | 0.7333 |
| nlpai-lab/KoE5 | 0.7756 | 0.8980 | 0.8434 | 0.8001 | 0.8351 | 0.9425 | 0.2942 | 0.6235 | 0.5841 | 0.7329 |
| dragonkue/BGE-m3-ko | — | — | 0.8738 | 0.7959 | 0.8155 | 0.9503 | 0.3784 | 0.6833 | 0.6099 | 0.7296 |
| intfloat/multilingual-e5-small | — | — | 0.8007 | 0.7516 | 0.7367 | 0.9053 | — | 0.6124 | 0.5597 | 0.7277 |
| Snowflake/snowflake-arctic-embed-m-v2.0 | — | — | 0.8381 | 0.7148 | 0.7727 | 0.8746 | — | 0.5978 | 0.5121 | 0.7184 |
| SamilPwC-AXNode-GenAI/PwC-Embedding_expr | 0.7400 | 0.8825 | 0.7849 | 0.7976 | 0.8346 | 0.9167 | 0.2663 | 0.6321 | 0.5666 | 0.7135 |
| Qwen/Qwen3-Embedding-0.6B | 0.7247 | 0.8503 | 0.8240 | 0.7660 | 0.8029 | 0.9160 | 0.3910 | 0.6002 | 0.4899 | 0.7072 |
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
| jhgan/ko-sroberta-multitask | — | — | 0.5833 | 0.6510 | 0.6921 | 0.8164 | 0.2175 | 0.3670 | 0.2948 | 0.5174 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | — | — | 0.4230 | 0.4590 | 0.6741 | 0.7149 | — | 0.2568 | 0.1272 | 0.4425 |

Notes: NDCG@10 only. MultiLongDocRetrieval uses the `ko` subset (test split). Multilingual tasks select a Korean subset (e.g. `kor_Hang-kor_Hang` for BelebeleRetrieval). Average is over available cells only.
