"""Verify each cell in eval/results_summary.md against the source of truth.

For every (model, task) cell:
  expected = JSON value if available, else result_hf.md curated value, else None
Compare against the rendered cell. Also re-check Averages and sort order.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "eval" / "results"
CURATED_MD = ROOT / "eval" / "result_hf.md"
SUMMARY_MD = ROOT / "eval" / "results_summary.md"

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

EPS = 5e-5  # rounding tolerance for "0.7820" vs 0.78199...


def fmt(x):
    return f"{x:.4f}" if x is not None else "—"


# -- JSON loader (mirrors build_results_table.py) ----------------------------
def extract_ndcg10(json_path, prefer_subset=None, prefer_split=None):
    data = json.loads(json_path.read_text())
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


def get_json_score(model_dir, task):
    files = sorted(model_dir.rglob(f"{task}.json"))
    if not files:
        return None
    prefer_subset = "ko" if task == "MultiLongDocRetrieval" else None
    prefer_split = "test" if task == "MultiLongDocRetrieval" else None
    return extract_ndcg10(files[0], prefer_subset=prefer_subset, prefer_split=prefer_split)


def discover_models_from_fs(results_root):
    out = []
    for org_dir in sorted(results_root.iterdir()):
        if not org_dir.is_dir() or org_dir.name == "my_experiment_result":
            continue
        for model_dir in sorted(org_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            out.append((f"{org_dir.name}/{model_dir.name}", model_dir))
    return out


# -- Curated MD parser -------------------------------------------------------
def parse_curated_md(md_path):
    if not md_path.exists():
        return {}
    rows = [r.strip() for r in md_path.read_text().splitlines() if r.strip().startswith("|")]
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


# -- Summary MD parser -------------------------------------------------------
def parse_summary_md(md_path):
    rows = [r.rstrip() for r in md_path.read_text().splitlines() if r.startswith("|")]
    if len(rows) < 3:
        return [], []
    headers = [h.strip() for h in rows[0].strip("|").split("|")]
    out_rows = []
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        out_rows.append(cells)
    return headers, out_rows


# -- Main verification -------------------------------------------------------
def parse_cell(s):
    s = s.strip()
    if s == "—" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main():
    curated = parse_curated_md(CURATED_MD)
    fs_models = discover_models_from_fs(RESULTS_ROOT)
    fs_index = {mid: mdir for mid, mdir in fs_models}

    headers, summary_rows = parse_summary_md(SUMMARY_MD)
    print(f"Summary headers: {headers}")
    print(f"Summary rows:    {len(summary_rows)}")
    print()

    expected_avg_col = headers.index("Average") if "Average" in headers else None
    task_cols = {t: headers.index(t) for t in ALL_TASKS if t in headers}

    cell_errors = []  # (model, task, summary_val, expected_val, source)
    avg_errors = []
    sort_errors = []
    coverage = []  # rows with #cells filled

    prev_avg = None
    for row in summary_rows:
        model = row[0]
        # Recompute expected for each task.
        for task, ci in task_cols.items():
            sv = parse_cell(row[ci])
            jv = get_json_score(fs_index[model], task) if model in fs_index else None
            cv = curated.get(model, {}).get(task)
            if jv is not None:
                expected = jv
                source = "json"
            elif cv is not None:
                expected = cv
                source = "curated"
            else:
                expected = None
                source = "none"
            # Compare.
            if expected is None and sv is None:
                continue
            if expected is None and sv is not None:
                cell_errors.append((model, task, sv, None, source, "summary has value but expected None"))
                continue
            if expected is not None and sv is None:
                cell_errors.append((model, task, None, expected, source, "summary missing but expected has value"))
                continue
            if abs(sv - expected) > EPS:
                cell_errors.append((model, task, sv, expected, source, f"diff={sv - expected:.5f}"))

        # Recompute average.
        present = [parse_cell(row[ci]) for ci in task_cols.values()]
        present = [v for v in present if v is not None]
        coverage.append((model, len(present)))
        recomputed = (sum(present) / len(present)) if present else None
        sv_avg = parse_cell(row[expected_avg_col]) if expected_avg_col else None
        if recomputed is None and sv_avg is None:
            pass
        elif recomputed is None or sv_avg is None:
            avg_errors.append((model, sv_avg, recomputed, "one side None"))
        elif abs(sv_avg - recomputed) > EPS:
            avg_errors.append((model, sv_avg, recomputed, f"diff={sv_avg - recomputed:.5f}"))

        # Sort order.
        if prev_avg is not None and sv_avg is not None and sv_avg > prev_avg + EPS:
            sort_errors.append((model, sv_avg, prev_avg))
        if sv_avg is not None:
            prev_avg = sv_avg

    # Coverage check vs result_hf.md (anyone missing?)
    missing_from_summary = []
    for cur_model in curated:
        if not any(r[0] == cur_model for r in summary_rows):
            missing_from_summary.append(cur_model)

    # Report.
    print(f"Cell mismatches: {len(cell_errors)}")
    for model, task, sv, ev, src, note in cell_errors[:20]:
        print(f"  ! {model} | {task}: summary={sv} expected={ev} ({src}) — {note}")
    if len(cell_errors) > 20:
        print(f"  ... ({len(cell_errors) - 20} more)")
    print()
    print(f"Average mismatches: {len(avg_errors)}")
    for model, sv, ev, note in avg_errors:
        print(f"  ! {model}: summary_avg={sv} recomputed={ev} — {note}")
    print()
    print(f"Sort-order violations: {len(sort_errors)}")
    for model, cur, prev in sort_errors:
        print(f"  ! {model}: avg={cur} > prev row avg={prev}")
    print()
    print(f"Models in result_hf.md not in summary: {len(missing_from_summary)}")
    for m in missing_from_summary:
        print(f"  ! {m}")
    print()
    print("Per-row cell coverage (count of non-empty task cells):")
    for m, c in coverage:
        bar = "█" * c + "·" * (len(ALL_TASKS) - c)
        print(f"  {bar}  {c}/{len(ALL_TASKS)}  {m}")

    total_problems = len(cell_errors) + len(avg_errors) + len(sort_errors) + len(missing_from_summary)
    print()
    print(f"=== {'OK: no issues' if total_problems == 0 else f'{total_problems} ISSUES'} ===")


if __name__ == "__main__":
    main()
