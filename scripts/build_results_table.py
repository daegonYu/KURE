"""Build a single unified markdown table merging:
- /data/daegon/workspace/code/eval/KURE/eval/result_hf.md (curated NDCG@10 across 7 tasks)
- /data/daegon/workspace/code/eval/KURE/eval/results/ JSONs (authoritative)

Output: eval/results_summary.md with one big table covering all 9 tasks + Average,
sorted by Average desc. JSON values win when both sources have a cell; result_hf.md
values fill the gaps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "eval" / "results"
CURATED_MD = ROOT / "eval" / "result_hf.md"
OUT = ROOT / "eval" / "results_summary.md"

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


# ---------------------------------------------------------------------------
# JSON-side extraction
# ---------------------------------------------------------------------------
def extract_ndcg10(
    json_path: Path,
    prefer_subset: str | None = None,
    prefer_split: str | None = None,
) -> float | None:
    with json_path.open() as f:
        data = json.load(f)
    scores = data.get("scores", {})
    split_order = (
        [prefer_split] + [s for s in scores if s != prefer_split]
        if prefer_split and prefer_split in scores
        else list(scores)
    )
    candidates: list[dict] = []
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


def discover_models_from_fs(results_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for org_dir in sorted(results_root.iterdir()):
        if not org_dir.is_dir() or org_dir.name == "my_experiment_result":
            continue
        for model_dir in sorted(org_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            out.append((f"{org_dir.name}/{model_dir.name}", model_dir))
    return out


def find_task_json(model_dir: Path, task: str) -> Path | None:
    for p in sorted(model_dir.rglob(f"{task}.json")):
        return p
    return None


def get_json_score(model_dir: Path, task: str) -> float | None:
    p = find_task_json(model_dir, task)
    if not p:
        return None
    prefer_subset = "ko" if task == "MultiLongDocRetrieval" else None
    prefer_split = "test" if task == "MultiLongDocRetrieval" else None
    return extract_ndcg10(p, prefer_subset=prefer_subset, prefer_split=prefer_split)


# ---------------------------------------------------------------------------
# result_hf.md parsing (curated)
# ---------------------------------------------------------------------------
def parse_curated_md(md_path: Path) -> dict[str, dict[str, float]]:
    if not md_path.exists():
        return {}
    text = md_path.read_text(encoding="utf-8")
    rows = [r.strip() for r in text.splitlines() if r.strip().startswith("|")]
    if len(rows) < 3:
        return {}
    headers = [h.strip() for h in rows[0].strip("|").split("|")]
    out: dict[str, dict[str, float]] = {}
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        model = cells[0]
        scores: dict[str, float] = {}
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


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def fmt(x: float | None) -> str:
    return f"{x:.4f}" if x is not None else "—"


def main() -> None:
    curated = parse_curated_md(CURATED_MD)
    fs_models = discover_models_from_fs(RESULTS_ROOT)
    fs_index = {mid: mdir for mid, mdir in fs_models}

    # Union of model ids from both sources.
    all_models = sorted(set(curated.keys()) | set(fs_index.keys()))

    rows: list[tuple[str, dict[str, float | None]]] = []
    for mid in all_models:
        scores: dict[str, float | None] = {t: None for t in ALL_TASKS}
        # Start from curated.
        for t, v in curated.get(mid, {}).items():
            if t in scores:
                scores[t] = v
        # Override with JSON when available.
        if mid in fs_index:
            mdir = fs_index[mid]
            for t in ALL_TASKS:
                jv = get_json_score(mdir, t)
                if jv is not None:
                    scores[t] = jv
        if any(v is not None for v in scores.values()):
            rows.append((mid, scores))

    # Sort by average desc.
    def avg_of(scores: dict[str, float | None]) -> float:
        vals = [v for v in scores.values() if v is not None]
        return sum(vals) / len(vals) if vals else -1.0

    rows.sort(key=lambda r: -avg_of(r[1]))

    # Render markdown.
    lines: list[str] = []
    lines.append("# MTEB Korean Retrieval NDCG@10 Summary")
    lines.append("")
    lines.append(
        f"Unified table merging `eval/result_hf.md` (curated) and "
        f"`eval/results/**/*.json` (authoritative). JSON values win on overlap; "
        f"curated values fill the rest."
    )
    lines.append("")
    header = ["Model"] + ALL_TASKS + ["Average"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for mid, scores in rows:
        vals = [scores[t] for t in ALL_TASKS]
        avg_vals = [v for v in vals if v is not None]
        avg = sum(avg_vals) / len(avg_vals) if avg_vals else None
        row = [mid] + [fmt(v) for v in vals] + [fmt(avg)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        "Notes: NDCG@10 only. MultiLongDocRetrieval uses the `ko` subset (test split). "
        "Multilingual tasks select a Korean subset (e.g. `kor_Hang-kor_Hang` for "
        "BelebeleRetrieval). Average is over available cells only."
    )
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Models in unified table: {len(rows)}")


if __name__ == "__main__":
    main()
