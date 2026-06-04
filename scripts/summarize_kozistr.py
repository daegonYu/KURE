"""Summarize MTEB Korean retrieval results for kozistr models + bge-m3 baseline.

Reads JSON outputs under eval/results/ and prints a markdown table of NDCG@10
for the 7 evaluated tasks plus the per-model average. Also prints Recall@10
and NDCG@1 tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"

# (display_name, glob pattern to find <task>.json files)
MODELS = [
    ("BAAI/bge-m3 (baseline)", RESULTS / "BAAI/bge-m3/BAAI__bge-m3/no_revision_available"),
    ("kozistr/ko_embed_v1",    RESULTS / "kozistr/ko_embed_v1/no_model_name__available/no_revision_available"),
    ("kozistr/ko_embed_v2",    RESULTS / "kozistr/ko_embed_v2/__data__models--kozistr--ko_embed_v2__snapshots__b9ed28facafc46caec22d1b5d1178dd5235315d4/no_revision_available"),
    ("kozistr/multi-emb-unsup-v5", RESULTS / "kozistr/multi-emb-unsup-v5/no_model_name__available/no_revision_available"),
]

TASKS = [
    "LawIRKo",
    "SQuADKorV1Retrieval",
    "AutoRAGRetrieval",
    "Ko-StrategyQA",
    "PublicHealthQA",
    "BelebeleRetrieval",
    "XPQARetrieval",
]


def load_scores(json_path: Path) -> dict:
    """Return scores from whichever split the task ships (test/dev/...).
    For multilingual tasks with multiple hf_subset entries, average the
    float metrics across subsets."""
    data = json.loads(json_path.read_text())
    scores = data["scores"]
    # Prefer test > dev > train > anything else.
    for preferred in ("test", "dev", "train"):
        if preferred in scores and scores[preferred]:
            test = scores[preferred]
            break
    else:
        only = next(iter(scores.values()), [])
        test = only
    if not test:
        return {}
    # Each entry corresponds to an hf_subset. Average the float-valued metrics.
    if len(test) == 1:
        return test[0]
    keys = set()
    for entry in test:
        keys.update(k for k, v in entry.items() if isinstance(v, (int, float)))
    out: dict = {}
    for k in keys:
        out[k] = mean(entry[k] for entry in test if k in entry)
    return out


def table(metric_key: str, label: str) -> str:
    header = "| Model | " + " | ".join(TASKS) + " | **Avg** |"
    sep = "|" + "---|" * (len(TASKS) + 2)
    rows = [f"### {label}", header, sep]
    for display_name, base_path in MODELS:
        vals: list[float | None] = []
        for task in TASKS:
            p = base_path / f"{task}.json"
            if not p.exists():
                vals.append(None)
                continue
            s = load_scores(p)
            v = s.get(metric_key)
            vals.append(float(v) if v is not None else None)
        formatted = [f"{v:.4f}" if v is not None else "—" for v in vals]
        finite = [v for v in vals if v is not None]
        avg = f"{mean(finite):.4f}" if finite else "—"
        rows.append("| " + display_name + " | " + " | ".join(formatted) + f" | **{avg}** |")
    return "\n".join(rows)


def main() -> None:
    parts = [
        "# kozistr 임베딩 모델 평가 결과 (MTEB Korean retrieval)",
        "",
        f"- 평가 태스크: {', '.join(TASKS)}",
        "- 제외: MIRACLRetrieval, MrTidyRetrieval (corpus 규모 큼)",
        "- ko_embed_v1, multi-emb-unsup-v5는 modules.json 부재 → CLS pooling + Normalize 명시 적용",
        "- ko_embed_v2는 sentence-transformers 표준 레이아웃 (CLS + Normalize) 그대로 로드",
        "",
        table("ndcg_at_10", "NDCG@10"),
        "",
        table("ndcg_at_1", "NDCG@1"),
        "",
        table("recall_at_10", "Recall@10"),
        "",
        table("main_score", "main_score (= NDCG@10 in retrieval)"),
    ]
    out = "\n".join(parts)
    print(out)
    target = ROOT / "eval" / "results_kozistr.md"
    target.write_text(out + "\n")
    print(f"\nWrote {target}", file=sys.stderr)


if __name__ == "__main__":
    main()
