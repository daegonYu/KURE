# MTEB Korean Retrieval NDCG@10 Summary

Unified table merging `eval/result_hf.md` (curated) and `eval/results/**/*.json` (authoritative). JSON values win on overlap; curated values fill the rest.

## 평가 데이터셋 (9개 task)

모든 task는 NDCG@10으로 측정되며, 다국어 task는 한국어 subset을 선택했습니다.

| Task | 도메인 / 형태 | 설명 |
|---|---|---|
| **LawIRKo** | 법률 / 한국어 | 한국 법률 도메인 정보 검색. 법률 질의에 적합한 조문·판례 문서를 찾는 task. |
| **SQuADKorV1Retrieval** | 일반 / 한국어 | 한국어 SQuAD v1 기반. 질문이 주어졌을 때 정답이 포함된 위키피디아 문단을 retrieval. |
| **AutoRAGRetrieval** | 다도메인 / 한국어 | AutoRAG 벤치마크의 한국어 retrieval. 다양한 도메인의 QA 컨텍스트 검색. |
| **Ko-StrategyQA** | 추론 / 한국어 | StrategyQA의 한국어판. 다단계 전략적 추론을 요하는 yes/no 질의에 대한 근거 문서 검색. |
| **PublicHealthQA** | 의료 / 한국어 | 한국 공중보건·의료 도메인 QA의 근거 문서 검색. |
| **BelebeleRetrieval** | 독해 / multilingual | Belebele MRC 데이터를 retrieval로 변환. 한국어 subset 3개(kor-kor, kor-eng, eng-kor)의 평균이 아닌 kor-kor 우선 사용. |
| **MultiLongDocRetrieval** | 장문 / multilingual | MLDR 다국어 long-document retrieval. `ko` subset(한국어 long-document) 사용 (test split). |
| **MIRACLRetrieval** | 위키 / multilingual | Wikipedia 기반 다국어 retrieval. 한국어 subset 사용. |
| **MrTidyRetrieval** | 위키 / multilingual | Mr. TyDi 한국어 subset, Wikipedia 기반 단답형 QA의 정답 문단 검색. |

## 평가 모델 (12개)

| Model | 설명 |
|---|---|
| **upstage/solar-embedding-1-large** | **Upstage**(한국 AI 스타트업) Solar 시리즈, 4096차원, REST API 기반 임베딩. |
| **dragonkue/snowflake-arctic-embed-l-v2.0-ko** | Hugging Face 사용자 **`dragonkue`**의 커뮤니티 fine-tune. Snowflake Arctic Embed L v2.0(XLM-R Large 기반)에 한국어 코퍼스로 추가 학습. |
| **nlpai-lab/KURE-v1** | **고려대학교 NLP & AI 연구실(nlpai-lab)** 공개. Korean Universal Retrieval Embedding, 한국어 retrieval 특화 학습. |
| **Snowflake/snowflake-arctic-embed-l-v2.0** | **Snowflake**(미국 데이터 클라우드 기업) Arctic 시리즈 다국어 retrieval embedding. XLM-RoBERTa Large 기반. |
| **BAAI/bge-m3** | **BAAI(Beijing Academy of Artificial Intelligence, 베이징인공지능연구원)** BGE-M3. multi-functionality(dense/sparse/multi-vector) · multi-linguality(100+ languages) 지원. |
| **intfloat/multilingual-e5-large** | **Microsoft**(intfloat 계정) Multilingual E5 large. XLM-RoBERTa Large 기반 contrastive 학습. |
| **nlpai-lab/KoE5** | **고려대학교 NLP & AI 연구실(nlpai-lab)** 공개. E5 계열을 한국어로 학습. |
| **SamilPwC-AXNode-GenAI/PwC-Embedding_expr** | **삼일PwC(Samil PwC) AXNode GenAI 팀** 실험용 임베딩 모델. |
| **Qwen/Qwen3-Embedding-0.6B** | **Alibaba Cloud Qwen 팀** Qwen3 Embedding 0.6B 파라미터 모델. |
| **telepix/PIXIE-Rune-Preview** | **TelePIX** PIXIE-Rune Preview (개발 중 버전). |
| **telepix/PIXIE-Rune-v1.0** | **TelePIX**(한국 TELEPIX, AI 솔루션 기업) PIXIE-Rune v1.0 임베딩. |
| **telepix/PIXIE-Rune-v1.5** | **TelePIX** PIXIE-Rune v1.5 임베딩. |

## 결과 표

> **모델 체급 주의사항**: `upstage/solar-embedding-1-large`는 임베딩 출력 차원이 **4096**으로, 1B 미만 모델들과 **동체급 비교가 아닙니다**. 1B 미만 모델은 일반적으로 1024차원(예: Snowflake Arctic Embed L v2.0, BGE-M3 등 XLM-R Large 기반)이며, **Qwen Embedding 8B가 4096차원**임을 고려할 때 upstage solar-embedding-1-large는 **1B 이상의 모델로 추정**됩니다. 따라서 이 표의 상위 점수가 더 큰 파라미터/차원으로부터 비롯될 수 있음을 감안해야 합니다.

| Model | LawIRKo | SQuADKorV1Retrieval | AutoRAGRetrieval | Ko-StrategyQA | PublicHealthQA | BelebeleRetrieval | MultiLongDocRetrieval | MIRACLRetrieval | MrTidyRetrieval | Average |
|---|---|---|---|---|---|---|---|---|---|---|
| upstage/solar-embedding-1-large | 0.7557 | 0.9521 | 0.8833 | 0.8366 | 0.8787 | 0.9684 | 0.3850 | 0.6703 | 0.5766 | 0.7674 |
| dragonkue/snowflake-arctic-embed-l-v2.0-ko | 0.7735 | 0.9447 | 0.9093 | 0.8050 | 0.8337 | 0.9518 | 0.4150 | 0.6685 | 0.5712 | 0.7636 |
| nlpai-lab/KURE-v1 | 0.7426 | 0.9357 | 0.8708 | 0.7999 | 0.8193 | 0.9502 | 0.4521 | 0.6816 | 0.5909 | 0.7603 |
| telepix/PIXIE-Rune-v1.5 | 0.7705 | 0.9457 | 0.8927 | 0.8064 | 0.8426 | 0.9617 | 0.4340 | 0.6393 | 0.5492 | 0.7602 |
| BAAI/bge-m3 | 0.7174 | 0.9038 | 0.8301 | 0.7941 | 0.8041 | 0.9316 | 0.4273 | 0.7015 | 0.6471 | 0.7508 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 0.7578 | 0.9121 | 0.8386 | 0.8045 | 0.8168 | 0.9271 | 0.3688 | 0.6608 | 0.5907 | 0.7419 |
| intfloat/multilingual-e5-large | 0.7293 | 0.9056 | 0.8134 | 0.8035 | 0.8253 | 0.9450 | 0.2708 | 0.6649 | 0.6421 | 0.7333 |
| nlpai-lab/KoE5 | 0.7756 | 0.8980 | 0.8434 | 0.8001 | 0.8351 | 0.9425 | 0.2942 | 0.6235 | 0.5841 | 0.7329 |
| SamilPwC-AXNode-GenAI/PwC-Embedding_expr | 0.7400 | 0.8825 | 0.7849 | 0.7976 | 0.8346 | 0.9167 | 0.2663 | 0.6321 | 0.5666 | 0.7135 |
| Qwen/Qwen3-Embedding-0.6B | 0.7247 | 0.8503 | 0.8240 | 0.7660 | 0.8029 | 0.9160 | 0.3910 | 0.6002 | 0.4899 | 0.7072 |
