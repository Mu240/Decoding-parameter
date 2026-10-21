"""Download TruthfulQA once, take a seeded category-stratified sample, and save it as JSONL.

Fixes vs the original script:
  * answer lists are stored as real JSON lists (no CSV round trip that turns arrays into strings)
  * ids are stable (they come from the dataset's own row index, not the post-sample row number)
  * the dataset revision, sampling seed and library versions are written to a manifest

Usage:
  python download_datasets.py --n 200 --seed 0 [--revision <commit-sha>]
  python download_datasets.py --fixture            # tiny synthetic file, no network (tests/dry runs)
After the first download, set HF_DATASETS_OFFLINE=1 for fully offline runs.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from config import DATA_DIR
from data_builders import make_tqa_fixture
from io_utils import write_json


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Round-robin over categories (shuffled with a fixed seed) until n rows are picked."""
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    cats = sorted(by_cat)
    for c in cats:
        rng.shuffle(by_cat[c])
    picked, i = [], 0
    while len(picked) < min(n, len(rows)):
        for c in cats:
            if i < len(by_cat[c]) and len(picked) < n:
                picked.append(by_cat[c][i])
        i += 1
    return sorted(picked, key=lambda r: r["id"])


def build_rows(gen, mc) -> list[dict]:
    mc_by_q = {m["question"]: m for m in mc}
    rows = []
    for idx, g in enumerate(gen):
        m = mc_by_q.get(g["question"])
        if m is None:
            continue
        rows.append({
            "id": f"tqa-{idx:04d}", "question": g["question"], "category": g["category"],
            "best_answer": g["best_answer"],
            "correct_answers": [str(x) for x in g["correct_answers"]],
            "incorrect_answers": [str(x) for x in g["incorrect_answers"]],
            "mc1_choices": list(m["mc1_targets"]["choices"]),
            "mc1_labels": [int(x) for x in m["mc1_targets"]["labels"]],
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--revision", default=None, help="pin a dataset revision (commit sha)")
    ap.add_argument("--out", default=str(DATA_DIR))
    ap.add_argument("--fixture", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.fixture:
        make_tqa_fixture(out / "truthfulqa_sample.jsonl", n=min(a.n, 60), seed=a.seed)
        write_json(out / "manifest.json", {"source": "synthetic fixture", "n": min(a.n, 60)})
        print("wrote synthetic fixture (NOT real data)")
        return

    import datasets
    kw = {"revision": a.revision} if a.revision else {}
    gen = datasets.load_dataset("truthfulqa/truthful_qa", "generation", **kw)["validation"]
    mc = datasets.load_dataset("truthfulqa/truthful_qa", "multiple_choice", **kw)["validation"]
    rows = build_rows(list(gen), list(mc))
    sample = stratified_sample(rows, a.n, a.seed)
    with open(out / "truthfulqa_sample.jsonl", "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out / "manifest.json", {
        "dataset": "truthfulqa/truthful_qa", "revision": a.revision or "unpinned (latest)",
        "n_total_joined": len(rows), "n_sample": len(sample), "seed": a.seed,
        "datasets_version": datasets.__version__,
    })
    print(f"wrote {len(sample)} questions to {out/'truthfulqa_sample.jsonl'}")


if __name__ == "__main__":
    main()
