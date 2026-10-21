"""Score all results, produce summary tables (CSV), paired contrasts, GEE models and plots.

  python analysis.py all
  python analysis.py export-labels --n 100        # sample answers for hand labelling
  python analysis.py kappa --file results/labels_to_fill.csv
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from data_builders import load_banks
from io_utils import read_jsonl
from scorers import (determinism_metrics, extract_answer, repeated_ngram_ratio, score_fake, score_json, score_mc,
                     score_truthfulqa)
from stats_utils import cluster_mean_ci, cohen_kappa, collapse_point, gee_logit, holm, paired_cluster_bootstrap

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


# ------------------------------------------------------------------------------------ loading / scoring
def load_results(experiment: str, results_dir) -> pd.DataFrame:
    rows = [r for r in read_jsonl(Path(results_dir) / f"{experiment}.jsonl")]
    n_err = sum(1 for r in rows if r.get("error"))
    rows = [r for r in rows if not r.get("error")]
    if n_err:
        print(f"  note: {experiment}: ignored {n_err} error rows")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates("key", keep="last").reset_index(drop=True)


def score_tqa_gen(df, bank, judge_path=None):
    recs = []
    for _, r in df.iterrows():
        item = bank[r["item_id"]]
        s = score_truthfulqa(r["output"], item["correct_answers"], item["incorrect_answers"])
        recs.append({"label": s["label"], "truthful": int(s["label"] == "truthful"), "false": int(s["label"] == "false"),
                     "abstain": int(s["label"] == "abstain"), "unclear": int(s["label"] == "unclear"),
                     "truthful_or_abstain": int(s["label"] in ("truthful", "abstain")), "words": s["words"],
                     "truncated": int(r.get("done_reason") == "length"), "category": item["category"]})
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    if judge_path and Path(judge_path).exists():
        j = pd.DataFrame(read_jsonl(judge_path)).drop_duplicates("key", keep="last")
        out = out.merge(j[["key", "judge_grade"]], on="key", how="left")
        out["judge_correct"] = np.where(out["judge_grade"].isna(), np.nan, (out["judge_grade"] == "correct").astype(float))
    return out


def score_tqa_mc(df):
    recs = [score_mc(r["output"], r["meta"]["correct_idx"], r["meta"]["n_choices"]) for _, r in df.iterrows()]
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)


def score_fake_df(df):
    recs = [score_fake(r["output"], r["meta"]["category"]) for _, r in df.iterrows()]
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    out["category"] = [m["category"] for m in df["meta"]]
    out["is_real"] = (out["category"] == "real_control").astype(int)
    return out


def score_json_df(df, bank):
    recs = []
    for _, r in df.iterrows():
        t = bank[r["item_id"]]
        recs.append(score_json(r["output"], t["schema"], t["gold"], r.get("done_reason")))
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    out["kind"] = [m["kind"] for m in df["meta"]]
    return out


def score_determinism_df(df, bank):
    recs = []
    for (model, cfg, sm, item), g in df.groupby(["model", "config", "seed_mode", "item_id"]):
        g = g.sort_values("rep")
        outs = g["output"].tolist()
        kind = bank[item]["answer_kind"]
        answers = [extract_answer(o, kind) for o in outs] if kind else None
        m = determinism_metrics(outs, answers)
        m.update({"model": model, "config": cfg, "seed_mode": sm, "item_id": item, "kind": bank[item]["kind"],
                  "truncated_frac": float((g["done_reason"] == "length").mean()),
                  "repeat_3gram": float(np.mean([repeated_ngram_ratio(o) for o in outs]))})
        recs.append(m)
    return pd.DataFrame(recs)


# ------------------------------------------------------------------------------------ summaries
def summarize(df, group_cols, metrics, cluster, n_boot=1000):
    recs = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        for m in metrics:
            if m not in g or g[m].isna().all():
                continue
            mean, lo, hi, n = cluster_mean_ci(g, m, cluster, n_boot=n_boot)
            row.update({m: mean, f"{m}_lo": lo, f"{m}_hi": hi})
            row["n_clusters"] = n
        recs.append(row)
    return pd.DataFrame(recs)


def contrasts(df, metric, cluster, group_cols=("model", "template")):
    """Each config vs its designated baseline (see config.py), paired by question, Holm-corrected."""
    recs = []
    for keys, g in df.groupby(list(group_cols)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        per = g.groupby(["config", cluster])[metric].mean().unstack(0)
        for name, cfg in C.CONFIGS.items():
            base = cfg["baseline"]
            if not base or name not in per.columns or base not in per.columns:
                continue
            d = (per[name] - per[base]).dropna().to_numpy()
            diff, lo, hi, p = paired_cluster_bootstrap(d)
            recs.append({**dict(zip(group_cols, keys)), "config": name, "baseline": base, "metric": metric,
                         "diff": diff, "ci_lo": lo, "ci_hi": hi, "p": p, "n_pairs": len(d)})
    out = pd.DataFrame(recs)
    if not out.empty:
        out["p_holm"] = holm(out["p"].to_numpy())
    return out


def run_gee(df, outcome, cluster):
    """outcome ~ config + model (+ template) with question-clustered SEs; returns a tidy table or a failure note."""
    try:
        rhs = 'C(config, Treatment(reference="T0.0"))'
        if df["model"].nunique() > 1:
            rhs += " + C(model)"
        if "template" in df and df["template"].nunique() > 1:
            rhs += " + C(template)"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return gee_logit(df, outcome, rhs, cluster)
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame([{"term": f"GEE failed: {type(e).__name__}: {e}"}])


# ------------------------------------------------------------------------------------ plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                         "axes.grid": True, "grid.alpha": 0.25})
    return plt


def plot_vs_temperature(summary, metric, ylabel, path, title):
    plt = _plt()
    temps = {n: c["temperature"] for n, c in C.CONFIGS.items() if c["group"] == "temperature"}
    s = summary[summary["config"].isin(temps)].copy()
    if s.empty:
        return
    s["T"] = s["config"].map(temps)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for i, (model, g) in enumerate(s.groupby("model")):
        g = g.groupby("T", as_index=False).mean(numeric_only=True).sort_values("T")
        ax.errorbar(g["T"], g[metric], yerr=[g[metric] - g[f"{metric}_lo"], g[f"{metric}_hi"] - g[metric]],
                    marker="o", capsize=3, color=PALETTE[i % len(PALETTE)], label=model)
    ax.set_xlabel("temperature (top-k, top-p neutral)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars_by_config(summary, metric, ylabel, path, title, hue=None, order=None):
    """Grouped bars: x = config, one bar per model (or per `hue` value)."""
    plt = _plt()
    if summary.empty:
        return
    hue = hue or "model"
    cfgs = [c for c in (order or list(C.CONFIGS)) if c in set(summary["config"])]
    levels = sorted(summary[hue].unique())
    fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(cfgs) * max(1, len(levels)) / 2 + 3), 4))
    w = 0.8 / len(levels)
    for i, lv in enumerate(levels):
        g = summary[summary[hue] == lv].groupby("config").mean(numeric_only=True).reindex(cfgs)
        x = np.arange(len(cfgs)) + i * w
        err = None
        if f"{metric}_lo" in g:
            err = [(g[metric] - g[f"{metric}_lo"]).fillna(0), (g[f"{metric}_hi"] - g[metric]).fillna(0)]
        ax.bar(x, g[metric], w, yerr=err, capsize=2, color=PALETTE[i % len(PALETTE)], label=str(lv))
    ax.set_xticks(np.arange(len(cfgs)) + w * (len(levels) - 1) / 2)
    ax.set_xticklabels(cfgs, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_frontier(det_summary, tqa_summary, path):
    """Diversity (unique-output ratio, varied seeds) vs truthfulness, per config. Only configs measured in both."""
    plt = _plt()
    d = det_summary[det_summary["seed_mode"] == "varied"][["model", "config", "unique_ratio"]]
    m = d.merge(tqa_summary[["model", "config", "truthful"]].groupby(["model", "config"], as_index=False).mean(),
                on=["model", "config"])
    if m.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, (model, g) in enumerate(m.groupby("model")):
        ax.scatter(g["unique_ratio"], g["truthful"], color=PALETTE[i % len(PALETTE)], label=model)
        for _, r in g.iterrows():
            ax.annotate(r["config"], (r["unique_ratio"], r["truthful"]), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
    ax.set_xlabel("output diversity (unique-output ratio, varied seeds)")
    ax.set_ylabel("truthful rate (TruthfulQA, heuristic scorer)")
    ax.set_title("Quality vs diversity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------------ orchestration
def analyze(results_dir, data_dir, out_dir, plots_dir):
    results_dir, out_dir, plots_dir = Path(results_dir), Path(out_dir), Path(plots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    banks = load_banks(data_dir)
    produced = {}
    tqa_sum = det_sum = None

    df = load_results("tqa_gen", results_dir)
    if not df.empty:
        s = score_tqa_gen(df, banks["tqa"], results_dir / "judge_tqa.jsonl")
        metrics = ["truthful", "false", "abstain", "unclear", "truthful_or_abstain", "words", "truncated"]
        if "judge_correct" in s:
            metrics.append("judge_correct")
        tqa_sum = summarize(s, ["model", "template", "config"], metrics, "item_id")
        tqa_sum.to_csv(out_dir / "tqa_gen_summary.csv", index=False)
        contr = contrasts(s, "truthful", "item_id")
        contr.to_csv(out_dir / "tqa_gen_contrasts.csv", index=False)
        run_gee(s, "truthful", "item_id").to_csv(out_dir / "tqa_gen_gee.csv", index=False)
        # per-model collapse temperature on the pure temperature sweep
        recs = []
        for model, g in tqa_sum[tqa_sum["config"].isin([n for n, c in C.CONFIGS.items() if c["group"] == "temperature"])].groupby("model"):
            g = g.groupby("config", as_index=False).mean(numeric_only=True)
            xs = [C.CONFIGS[c]["temperature"] for c in g["config"]]
            recs.append({"model": model, "collapse_T": collapse_point(xs, g["truthful"].tolist(), 0.0, 0.05)})
        pd.DataFrame(recs).to_csv(out_dir / "tqa_gen_collapse_temperature.csv", index=False)
        plot_vs_temperature(tqa_sum, "truthful", "truthful rate", plots_dir / "tqa_truthful_vs_temperature.png",
                            "TruthfulQA truthful rate vs temperature")
        plot_bars_by_config(tqa_sum, "truthful", "truthful rate", plots_dir / "tqa_truthful_by_config.png",
                            "TruthfulQA truthful rate by config")
        produced["tqa_gen"] = len(s)

    df = load_results("tqa_mc", results_dir)
    if not df.empty:
        s = score_tqa_mc(df)
        mc_sum = summarize(s, ["model", "config"], ["correct", "parsed"], "item_id")
        mc_sum.to_csv(out_dir / "tqa_mc_summary.csv", index=False)
        contrasts(s.assign(template="plain"), "correct", "item_id").to_csv(out_dir / "tqa_mc_contrasts.csv", index=False)
        produced["tqa_mc"] = len(s)

    df = load_results("fake", results_dir)
    if not df.empty:
        s = score_fake_df(df)
        fk = summarize(s[s["is_real"] == 0], ["model", "template", "config"], ["fabricated", "declined"], "item_id")
        fk.to_csv(out_dir / "fake_summary.csv", index=False)
        fr = summarize(s[s["is_real"] == 1].assign(refused=lambda x: x["declined"]), ["model", "template", "config"],
                       ["refused"], "item_id")
        fr.to_csv(out_dir / "real_control_false_refusal.csv", index=False)
        contrasts(s[s["is_real"] == 0], "fabricated", "item_id").to_csv(out_dir / "fake_contrasts.csv", index=False)
        plot_bars_by_config(fk, "fabricated", "fabrication rate", plots_dir / "fabrication_by_config.png",
                            "Fabrication rate on nonexistent entities")
        produced["fake"] = len(s)

    df = load_results("determinism", results_dir)
    if not df.empty:
        d = score_determinism_df(df, banks["det"])
        d.to_csv(out_dir / "determinism_per_prompt.csv", index=False)
        cols = ["modal_rate", "unique_ratio", "mean_pairwise_edit", "first_divergence_frac", "tar_raw", "tar_answer",
                "truncated_frac", "repeat_3gram"]
        det_sum = summarize(d, ["model", "config", "seed_mode"], cols, "item_id")
        det_sum.to_csv(out_dir / "determinism_summary.csv", index=False)
        summarize(d, ["model", "config", "seed_mode", "kind"], ["modal_rate", "unique_ratio", "tar_raw", "tar_answer"],
                  "item_id").to_csv(out_dir / "determinism_by_prompt_kind.csv", index=False)
        sub = det_sum.rename(columns={"seed_mode": "seed regime"})
        plot_bars_by_config(sub, "modal_rate", "modal-output rate (1 = fully repeatable)",
                            plots_dir / "determinism_by_config.png", "Repeatability by config and seed regime",
                            hue="seed regime", order=C.DETERMINISM_CONFIGS)
        produced["determinism"] = len(df)

    df = load_results("json", results_dir)
    if not df.empty:
        s = score_json_df(df, banks["json"])
        js = summarize(s, ["model", "mode", "config"], ["strict_parse", "schema_ok_strict", "schema_ok_lenient",
                                                        "field_acc", "truncated"], "item_id")
        js.to_csv(out_dir / "json_summary.csv", index=False)
        tax = s.groupby(["model", "mode", "config", "category"]).size().rename("count").reset_index()
        tax.to_csv(out_dir / "json_failure_taxonomy.csv", index=False)
        err_rows = []
        for _, r in s.iterrows():
            for e in r["schema_errors"]:
                err_rows.append({"model": r["model"], "mode": r["mode"], "config": r["config"], "schema_error": e})
        pd.DataFrame(err_rows, columns=["model", "mode", "config", "schema_error"]).groupby(
            ["model", "mode", "config", "schema_error"]).size().rename("count").reset_index().to_csv(
            out_dir / "json_schema_errors.csv", index=False)
        plot_bars_by_config(js, "schema_ok_strict", "strictly valid (json.loads + schema)", plots_dir / "json_validity.png",
                            "JSON validity: prompt-only vs schema-constrained", hue="mode", order=C.JSON_CONFIGS)
        produced["json"] = len(s)

    if tqa_sum is not None and det_sum is not None:
        plot_frontier(det_sum, tqa_sum, plots_dir / "quality_diversity_frontier.png")
    return produced


def export_labels(results_dir, data_dir, out_path, n, seed=0):
    banks = load_banks(data_dir)
    s = score_tqa_gen(load_results("tqa_gen", results_dir), banks["tqa"])
    s = s.sample(min(n, len(s)), random_state=seed)
    out = pd.DataFrame({"key": s["key"], "question": [banks["tqa"][i]["question"] for i in s["item_id"]],
                        "correct_refs": [" | ".join(banks["tqa"][i]["correct_answers"][:3]) for i in s["item_id"]],
                        "answer": s["output"], "auto_label": s["label"], "human_label": ""})
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows. Fill human_label with truthful / false / abstain / unclear, then run `kappa`.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "export-labels", "kappa"])
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--plots-dir", default=str(C.PLOTS_DIR))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--file", default=None)
    a = ap.parse_args()
    out_dir = a.out_dir or str(Path(a.results_dir) / "tables")
    if a.cmd == "all":
        print(analyze(a.results_dir, a.data_dir, out_dir, a.plots_dir))
        print(f"tables -> {out_dir}   plots -> {a.plots_dir}")
    elif a.cmd == "export-labels":
        export_labels(a.results_dir, a.data_dir, a.file or Path(a.results_dir) / "labels_to_fill.csv", a.n)
    else:
        d = pd.read_csv(a.file).dropna(subset=["human_label"])
        d = d[d["human_label"].astype(str).str.strip() != ""]
        print(f"n={len(d)} kappa={cohen_kappa(d['auto_label'], d['human_label']):.3f} "
              f"raw agreement={(d['auto_label'] == d['human_label']).mean():.3f}")


if __name__ == "__main__":
    main()
