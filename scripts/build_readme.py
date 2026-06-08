"""Generate README.md containing:
- Task descriptions (9 standard MTEB Korean retrieval tasks, MLDR excluded)
- Model descriptions (all non-upstage models with results, curated bullets)
- Results table (NDCG@10, all non-upstage models × 9 tasks, sorted by Average desc)
- Evaluation command at the bottom

Sources of truth:
- eval/results/**/*.json (authoritative)
- eval/result_hf.md (curated fallback for tasks not in JSON)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "eval" / "results"
CURATED_MD = ROOT / "eval" / "result_hf.md"
OUT = ROOT / "README.md"

ALL_TASKS = [
    "LawIRKo",
    "SQuADKorV1Retrieval",
    "AutoRAGRetrieval",
    "Ko-StrategyQA",
    "PublicHealthQA",
    "BelebeleRetrieval",
    "MultiLongDocRetrieval",
    "MIRACLRetrieval",
    "MrTidyRetrieval",
]

# Hand-curated descriptions. Models without an entry get a placeholder line.
MODEL_DESCRIPTIONS: dict[str, str] = {
    "dragonkue/snowflake-arctic-embed-l-v2.0-ko":
        "Hugging Face 사용자 **`dragonkue`**의 커뮤니티 fine-tune. Snowflake Arctic Embed L v2.0(XLM-R Large 기반)에 한국어 코퍼스로 추가 학습. 1024차원.",
    "nlpai-lab/KURE-v1":
        "**고려대학교 NLP & AI 연구실(nlpai-lab)** 공개. Korean Universal Retrieval Embedding, 한국어 retrieval 특화 학습.",
    "Snowflake/snowflake-arctic-embed-l-v2.0":
        "**Snowflake**(미국 데이터 클라우드 기업) Arctic 시리즈 다국어 retrieval embedding. XLM-RoBERTa Large 기반, 1024차원.",
    "Snowflake/snowflake-arctic-embed-m-v2.0":
        "**Snowflake** Arctic 시리즈 medium 변형. multilingual retrieval embedding.",
    "BAAI/bge-m3":
        "**BAAI(Beijing Academy of Artificial Intelligence, 베이징인공지능연구원)** BGE-M3. multi-functionality(dense/sparse/multi-vector) · multi-linguality(100+ languages) 지원.",
    "BAAI/bge-multilingual-gemma2":
        "**BAAI** Gemma2 기반 multilingual embedding. instruct-style query prefix 사용, last-token pooling, fp16.",
    "intfloat/multilingual-e5-large":
        "**Microsoft**(intfloat 계정) Multilingual E5 large. XLM-RoBERTa Large 기반 contrastive 학습.",
    "intfloat/multilingual-e5-large-instruct":
        "**Microsoft** Multilingual E5 large instruct 변형. instruction-aware contrastive 학습.",
    "intfloat/multilingual-e5-base":
        "**Microsoft** Multilingual E5 base. XLM-RoBERTa Base 기반.",
    "intfloat/multilingual-e5-small":
        "**Microsoft** Multilingual E5 small. 경량 base 모델.",
    "intfloat/e5-mistral-7b-instruct":
        "**Microsoft** E5 mistral 7B instruct. Mistral 7B LLM 기반 instruction-aware embedding.",
    "nlpai-lab/KoE5":
        "**고려대학교 NLP & AI 연구실(nlpai-lab)** 공개. E5 계열을 한국어로 학습한 모델.",
    "SamilPwC-AXNode-GenAI/PwC-Embedding_expr":
        "**삼일PwC(Samil PwC) AXNode GenAI 팀** 실험용 임베딩 모델.",
    "Qwen/Qwen3-Embedding-0.6B":
        "**Alibaba Cloud Qwen 팀** Qwen3 Embedding 0.6B 파라미터 모델.",
    "jinaai/jina-embeddings-v3":
        "**Jina AI**(독일/베를린, 검색·RAG 인프라 회사) v3 multilingual embedding. XLM-R 기반에 LoRA adapter로 task-specific fine-tune.",
    "jinaai/jina-embeddings-v5-text-small":
        "**Jina AI** v5 text small. 다중 어댑터(retrieval/classification/clustering/text-matching) 구조. (이번 평가에서 sentence-transformers 호환 이슈로 결과 미수집)",
    "telepix/PIXIE-Rune-v1.5":
        "**TelePIX**(한국 TELEPIX, AI 솔루션 기업) PIXIE-Rune v1.5 임베딩.",
    "telepix/PIXIE-Rune-v1.0":
        "**TelePIX** PIXIE-Rune v1.0 임베딩.",
    "telepix/PIXIE-Rune-Preview":
        "**TelePIX** PIXIE-Rune Preview (개발 중 버전).",
    "dragonkue/BGE-m3-ko":
        "Hugging Face 사용자 **`dragonkue`**의 커뮤니티 fine-tune. BGE-M3에 한국어 코퍼스 추가 학습.",
    "dragonkue/multilingual-e5-small-ko":
        "Hugging Face 사용자 **`dragonkue`**의 multilingual-e5-small 한국어 fine-tune.",
    "exp-models/dragonkue-KoEn-E5-Tiny":
        "Hugging Face 사용자 **`dragonkue`** 공개 KoEn-E5 Tiny. 한국어/영어 경량 E5.",
    "Alibaba-NLP/gte-Qwen2-7B-instruct":
        "**Alibaba NLP** GTE Qwen2 7B instruct. Qwen2 7B 기반 instruction-aware embedding.",
    "Alibaba-NLP/gte-multilingual-base":
        "**Alibaba NLP** GTE multilingual base. multilingual general-purpose embedding.",
    "nomic-ai/nomic-embed-text-v2-moe":
        "**Nomic AI**(미국, 임베딩·AI 검색 회사) nomic-embed-text v2 MoE. Mixture-of-Experts 기반 multilingual.",
    "openai/text-embedding-3-large":
        "**OpenAI** text-embedding-3-large. OpenAI 임베딩 API의 대형 모델.",
    "upskyy/bge-m3-korean":
        "Hugging Face 사용자 **`upskyy`**의 BGE-M3 한국어 fine-tune.",
    "Salesforce/SFR-Embedding-2_R":
        "**Salesforce Research** SFR-Embedding 2 R. Mistral 7B 기반 reranking 강화 embedding.",
    "ibm-granite/granite-embedding-278m-multilingual":
        "**IBM Research** Granite multilingual embedding 278M.",
    "ibm-granite/granite-embedding-107m-multilingual":
        "**IBM Research** Granite multilingual embedding 107M (경량).",
    "jhgan/ko-sroberta-multitask":
        "Hugging Face 사용자 **`jhgan`**의 한국어 SBERT/RoBERTa multitask 모델.",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
        "**UKP Lab / sentence-transformers 커뮤니티** Multilingual MiniLM-L12 v2. 경량 multilingual SBERT.",
    "nvidia/llama-nemotron-embed-vl-1b-v2":
        "**NVIDIA** llama-nemotron VL 1B v2. 멀티모달(VL) 임베더를 텍스트 전용으로 평가. 빌트인 e5 스타일 `query: `/`passage: ` prefix, max_seq_length=8192, dim 2048.",
    "local/en_ja_break_bench_data_v2_wkl8b_kl_only_tau005_bs256_lr1e-5":
        "**(내부 실험)** Snowflake Arctic Embed L v2.0 기반 EN-JA fine-tune (KL distill, τ=0.05, checkpoint-748). max_seq_length=8192로 평가.",
}

# README에서 제외할 모델 (exact id) / 제외할 org prefix.
SKIP_MODELS: set[str] = {"telepix/PIXIE-Rune-Preview"}
SKIP_ORG_PREFIXES: tuple[str, ...] = ("upstage/", "kozistr/")


# ---------------------------------------------------------------------------
# Data extraction (mirrors build_results_table.py)
# ---------------------------------------------------------------------------
def extract_ndcg10(
    p: Path,
    prefer_subset: str | None = None,
    prefer_split: str | None = None,
) -> float | None:
    data = json.loads(p.read_text())
    scores = data.get("scores", {})
    split_order = (
        [prefer_split] + [s for s in scores if s != prefer_split]
        if prefer_split and prefer_split in scores
        else list(scores)
    )
    candidates = []
    for s in split_order:
        for v in scores.get(s, []):
            if "ndcg_at_10" in v:
                candidates.append(v)
    if not candidates:
        return None
    if prefer_subset:
        for c in candidates:
            if c.get("hf_subset") == prefer_subset:
                return c["ndcg_at_10"]
    for c in candidates:
        langs = c.get("languages", [])
        if any("kor" in str(l).lower() for l in langs):
            return c["ndcg_at_10"]
    return candidates[0]["ndcg_at_10"]


def get_json_score(model_dir: Path, task: str) -> float | None:
    files = sorted(model_dir.rglob(f"{task}.json"))
    if not files:
        return None
    prefer_subset = "ko" if task == "MultiLongDocRetrieval" else None
    prefer_split = "test" if task == "MultiLongDocRetrieval" else None
    return extract_ndcg10(files[0], prefer_subset=prefer_subset, prefer_split=prefer_split)


def discover_models() -> list[tuple[str, Path]]:
    out = []
    for org_dir in sorted(RESULTS_ROOT.iterdir()):
        if not org_dir.is_dir() or org_dir.name in {"my_experiment_result", "upstage"}:
            continue
        for m in sorted(org_dir.iterdir()):
            if not m.is_dir():
                continue
            out.append((f"{org_dir.name}/{m.name}", m))
    return out


def parse_curated_md() -> dict[str, dict[str, float]]:
    if not CURATED_MD.exists():
        return {}
    rows = [r.strip() for r in CURATED_MD.read_text().splitlines() if r.strip().startswith("|")]
    if len(rows) < 3:
        return {}
    headers = [h.strip() for h in rows[0].strip("|").split("|")]
    out = {}
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        model = cells[0]
        scores = {}
        for h, c in zip(headers[1:], cells[1:]):
            if h == "Average":
                continue
            cleaned = re.sub(r"[*_`]", "", c).strip()
            if not cleaned or cleaned == "—":
                continue
            try:
                scores[h] = float(cleaned)
            except ValueError:
                pass
        out[model] = scores
    return out


def fmt(x):
    return f"{x:.4f}" if x is not None else "—"


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def main() -> None:
    curated = parse_curated_md()
    fs = discover_models()
    fs_index = {mid: mdir for mid, mdir in fs}
    all_models = sorted(set(curated.keys()) | set(fs_index.keys()))
    # Drop upstage + explicitly excluded models / orgs (see SKIP_* above).
    all_models = [
        m
        for m in all_models
        if m not in SKIP_MODELS and not m.startswith(SKIP_ORG_PREFIXES)
    ]

    rows = []
    for mid in all_models:
        scores = {t: None for t in ALL_TASKS}
        for t, v in curated.get(mid, {}).items():
            if t in scores:
                scores[t] = v
        if mid in fs_index:
            for t in ALL_TASKS:
                jv = get_json_score(fs_index[mid], t)
                if jv is not None:
                    scores[t] = jv
        if any(v is not None for v in scores.values()):
            rows.append((mid, scores))

    def avg_of(s):
        vals = [v for v in s.values() if v is not None]
        return sum(vals) / len(vals) if vals else -1.0

    rows.sort(key=lambda r: -avg_of(r[1]))

    L = []
    L.append("# MTEB Korean Retrieval Evaluation")
    L.append("")
    L.append(
        "한국어 retrieval 임베딩 모델을 MTEB Korean retrieval task로 NDCG@10 평가한 결과를 정리한 문서입니다. "
        "**Upstage solar-embedding-1-large**(임베딩 4096차원, 1B+ 추정)는 동체급 비교가 아니므로 본 표에서 제외했습니다."
    )
    L.append("")

    # Tasks
    L.append("## 평가 데이터셋 (9개 task)")
    L.append("")
    L.append("모든 task는 NDCG@10으로 측정하며, 다국어 task는 한국어 subset을 선택했습니다.")
    L.append("")
    L.append("| Task | 도메인 / 형태 | 설명 |")
    L.append("|---|---|---|")
    L.append("| **LawIRKo** | 법률 / 한국어 | 한국 법률 도메인 정보 검색. 법률 질의에 적합한 조문·판례 문서를 찾는 task. |")
    L.append("| **SQuADKorV1Retrieval** | 일반 / 한국어 | 한국어 SQuAD v1 기반. 질문이 주어졌을 때 정답이 포함된 위키피디아 문단을 retrieval. |")
    L.append("| **AutoRAGRetrieval** | 다도메인 / 한국어 | AutoRAG 벤치마크의 한국어 retrieval. 다양한 도메인의 QA 컨텍스트 검색. |")
    L.append("| **Ko-StrategyQA** | 추론 / 한국어 | StrategyQA의 한국어판. 다단계 전략적 추론을 요하는 yes/no 질의에 대한 근거 문서 검색. |")
    L.append("| **PublicHealthQA** | 의료 / 한국어 | 한국 공중보건·의료 도메인 QA의 근거 문서 검색. |")
    L.append("| **BelebeleRetrieval** | 독해 / multilingual | Belebele MRC 데이터를 retrieval로 변환. 한국어 subset 3개(kor-kor, kor-eng, eng-kor) 중 kor-kor 우선 사용. |")
    L.append("| **MultiLongDocRetrieval** | 장문 / multilingual | MLDR 다국어 long-document retrieval. `ko` subset(한국어 long-document) 사용. |")
    L.append("| **MIRACLRetrieval** | 위키 / multilingual | Wikipedia 기반 다국어 retrieval. 한국어 subset 사용. |")
    L.append("| **MrTidyRetrieval** | 위키 / multilingual | Mr. TyDi 한국어 subset, Wikipedia 기반 단답형 QA의 정답 문단 검색. |")
    L.append("")

    # Models
    L.append("## 평가 모델")
    L.append("")
    L.append(
        f"비교 대상: **{len(rows)}개 모델** (upstage 제외). "
        "회사/팀 정보와 핵심 특징만 요약했고, 상세 사양은 각 모델의 Hugging Face 카드를 참조하세요."
    )
    L.append("")
    L.append("| Model | 설명 |")
    L.append("|---|---|")
    for mid, _ in rows:
        desc = MODEL_DESCRIPTIONS.get(mid, "(설명 미작성)")
        L.append(f"| **{mid}** | {desc} |")
    L.append("")

    # Results
    L.append("## 결과 표 (NDCG@10)")
    L.append("")
    L.append(
        "Average 내림차순. 누락 셀(`—`)은 해당 (model, task) 평가 결과가 없는 경우이며 평균 계산에서 제외했습니다."
    )
    L.append("")
    header = ["Model"] + ALL_TASKS + ["Average"]
    L.append("| " + " | ".join(header) + " |")
    L.append("|" + "|".join(["---"] * len(header)) + "|")
    for mid, scores in rows:
        vals = [scores[t] for t in ALL_TASKS]
        avail = [v for v in vals if v is not None]
        avg = sum(avail) / len(avail) if avail else None
        cells = [mid] + [fmt(v) for v in vals] + [fmt(avg)]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # Evaluation command
    L.append("## 평가 실행")
    L.append("")
    L.append("`eval/evaluate.py`로 단일/다중 모델 × 단일/다중 task를 평가합니다. tmux 세션 내 실행을 권장합니다.")
    L.append("")
    L.append("### 환경 준비")
    L.append("")
    L.append("```bash")
    L.append("# uv 가상 환경 (.venv)")
    L.append("uv sync")
    L.append("# 일부 모델 추가 의존성")
    L.append('uv pip install einops peft  # jinaai-v3 / v5 family에 필요')
    L.append("```")
    L.append("")
    L.append("### 단일 GPU에서 다중 모델 평가")
    L.append("")
    L.append("```bash")
    L.append("cd eval")
    L.append("CUDA_VISIBLE_DEVICES=0 uv run evaluate.py \\")
    L.append("    --models 'BAAI/bge-m3,nlpai-lab/KURE-v1,Qwen/Qwen3-Embedding-0.6B' \\")
    L.append("    --tasks 'LawIRKo,SQuADKorV1Retrieval,AutoRAGRetrieval,Ko-StrategyQA,PublicHealthQA,BelebeleRetrieval,XPQARetrieval,MIRACLRetrieval,MrTidyRetrieval' \\")
    L.append("    --gpu 0")
    L.append("```")
    L.append("")
    L.append("- `--models`: 콤마로 구분된 모델 ID. Hugging Face 모델 ID 또는 로컬 경로. `upstage/<name>` 형식이면 API 호출.")
    L.append("- `--tasks`: 콤마로 구분된 MTEB task 이름. 위 9개가 표준 한국어 retrieval set.")
    L.append("- `--gpu`: 사용 GPU 번호.")
    L.append("- `--quantize`: 임베딩을 binary로 양자화 (선택).")
    L.append("- 결과는 `eval/results/<org>/<model>/<...>/<task>.json` 으로 저장됨. 같은 (model, task) 결과가 이미 있으면 mteb가 자동 skip.")
    L.append("")
    L.append("### 프리셋 스크립트")
    L.append("")
    L.append("```bash")
    L.append("# eval/eval.sh: default 또는 upstage 프로파일")
    L.append("cd eval")
    L.append("./eval.sh 0 default   # GPU 0, 기본 모델 묶음")
    L.append("./eval.sh 0 upstage   # API 기반 upstage 평가")
    L.append("```")
    L.append("")
    L.append("### 결과 표 재생성")
    L.append("")
    L.append("```bash")
    L.append("uv run python scripts/build_results_table.py     # eval/results_summary.md 갱신")
    L.append("uv run python scripts/verify_results_table.py    # 셀 정확성 검증")
    L.append("uv run python scripts/build_readme.py            # README.md 갱신 (이 파일)")
    L.append("```")
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Models in README: {len(rows)}")


if __name__ == "__main__":
    main()
