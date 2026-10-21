"""Experiment planning: turns item banks + configs into a flat list of generation jobs.

A job is a plain dict. Its `key` uniquely identifies the generation so an interrupted run can
resume by skipping keys already present in the results file.
"""
from __future__ import annotations

from collections import Counter

import config as C
from io_utils import stable_int

QA_PROMPT = C.QA_INSTRUCTION + "\n\nQuestion: {q}\nAnswer:"
MC_PROMPT = ("Answer the following multiple-choice question. Reply with only the letter of the correct option.\n\n"
             "Question: {q}\n{opts}\nAnswer:")


def make_key(job: dict) -> str:
    return "|".join([job["experiment"], job["model"], job["config"], job.get("template", "plain"),
                     job["item_id"], job.get("seed_mode", ""), str(job.get("rep", 0)), job.get("mode", "")])


def _job(experiment, model, cfg, item_id, prompt, *, template="plain", seed=None, seed_mode="", rep=0,
         mode="", fmt=None, hint=None, num_predict=None, meta=None):
    j = {"experiment": experiment, "model": model, "config": cfg["name"], "cfg": cfg, "item_id": item_id,
         "prompt": prompt, "template": template, "system": C.TEMPLATES.get(template), "seed": seed,
         "seed_mode": seed_mode, "rep": rep, "mode": mode, "fmt": fmt, "hint": hint,
         "num_predict": num_predict, "meta": meta or {}}
    j["key"] = make_key(j)
    return j


def plan_tqa_gen(models, config_names, questions, templates=("plain",)):
    jobs = []
    for m in models:
        for t in templates:
            for cn in config_names:
                cfg = C.CONFIGS[cn]
                for q in questions:
                    jobs.append(_job("tqa_gen", m, cfg, q["id"], QA_PROMPT.format(q=q["question"]), template=t,
                                     seed=stable_int(q["id"]), seed_mode="paired",
                                     hint={"kind": "qa", "ideal": q["best_answer"],
                                           "wrong": (q["incorrect_answers"] or ["Unknown."])[0]}))
    return jobs


def mc_layout(q: dict):
    """Deterministic shuffle of the MC1 options (HF lists the correct answer first)."""
    import random
    order = list(range(len(q["mc1_choices"])))
    random.Random(stable_int(q["id"] + "mc")).shuffle(order)
    correct_pos = order.index(q["mc1_labels"].index(1))
    return order, correct_pos


def plan_tqa_mc(models, config_names, questions):
    jobs = []
    for m in models:
        for cn in config_names:
            cfg = C.CONFIGS[cn]
            for q in questions:
                order, cpos = mc_layout(q)
                opts = "\n".join(f"{chr(65 + i)}. {q['mc1_choices'][k]}" for i, k in enumerate(order))
                jobs.append(_job("tqa_mc", m, cfg, q["id"], MC_PROMPT.format(q=q["question"], opts=opts),
                                 seed=stable_int(q["id"]), seed_mode="paired", num_predict=4,
                                 hint={"kind": "mc", "n_choices": len(order), "ideal_idx": cpos},
                                 meta={"n_choices": len(order), "correct_idx": cpos}))
    return jobs


def plan_fake(models, config_names, items, reps=C.FAKE_REPS, templates=("plain",)):
    jobs = []
    for m in models:
        for t in templates:
            for cn in config_names:
                cfg = C.CONFIGS[cn]
                for it in items:
                    for rep in range(reps):
                        if C.is_greedy(cfg) and rep > 0:
                            continue
                        kind = "real" if it["category"] == "real_control" else "fake"
                        jobs.append(_job("fake", m, cfg, it["id"], it["prompt"], template=t, seed=rep, seed_mode="rep",
                                         rep=rep, hint={"kind": kind}, meta={"category": it["category"]}))
    return jobs


def plan_determinism(models, config_names, prompts, reps=C.DETERMINISM_REPS, seed_modes=("fixed", "varied", "unseeded"),
                     num_predict=None):
    """Same prompt x N runs, under three seed regimes:
    fixed    -> same seed every run  (engine determinism: should be ~identical even at T>0)
    varied   -> seed = rep           (genuine sampling variability)
    unseeded -> no seed sent         (what a casual user gets)"""
    jobs = []
    for m in models:
        for cn in config_names:
            cfg = C.CONFIGS[cn]
            for p in prompts:
                for sm in seed_modes:
                    for rep in range(reps):
                        seed = C.DETERMINISM_SEEDS["fixed"] if sm == "fixed" else rep if sm == "varied" else None
                        jobs.append(_job("determinism", m, cfg, p["id"], p["prompt"], seed=seed, seed_mode=sm, rep=rep,
                                         num_predict=num_predict, hint={"kind": "free"}, meta={"kind": p["kind"]}))
    return jobs


def plan_json(models, config_names, tasks, reps=C.JSON_REPS, modes=("prompt", "constrained")):
    jobs = []
    for m in models:
        for cn in config_names:
            cfg = C.CONFIGS[cn]
            for t in tasks:
                for mode in modes:
                    for rep in range(reps):
                        if C.is_greedy(cfg) and rep > 0:
                            continue
                        jobs.append(_job("json", m, cfg, t["id"], t["prompt"], seed=rep, seed_mode="rep", rep=rep,
                                         mode=mode, fmt=t["schema"] if mode == "constrained" else None,
                                         num_predict=256, hint={"kind": "json", "ideal": t["gold"]},
                                         meta={"kind": t["kind"]}))
    return jobs


CONTROL_PROMPT_IDS = ["det-creative-0", "det-creative-1", "det-list-0", "det-list-1", "det-ambiguous-0"]


def plan_controls(models, prompts):
    """Sanity controls that tell you whether your options really reach the sampler."""
    by_id = {p["id"]: p for p in prompts}
    t0, k1, t10, t16 = (C.CONFIGS[n] for n in ("T0.0", "T1.0_k1", "T1.0", "T1.6"))
    k0 = C._c("T1.0_k0", 1.0, k=0, group="control")
    jobs = []
    for m in models:
        for pid in CONTROL_PROMPT_IDS:
            p = by_id[pid]
            add = lambda ctl, cfg, seed, sm, rep: jobs.append(_job(  # noqa: E731
                "controls", m, cfg, pid, p["prompt"], seed=seed, seed_mode=sm, rep=rep, mode=ctl, hint={"kind": "free"}))
            for sm, seed in (("unseeded", None), ("seed1", 1)):          # greedy twice, two ways
                for rep in (0, 1):
                    add("greedy_repeat", t0, seed, sm, rep)
            add("k1_vs_t0", k1, 7, "s7", 0)
            add("k1_vs_t0", t0, 7, "s7", 0)
            for s in range(5):
                add("varies_t16", t16, s, "varied", s)
            for rep in (0, 1):
                add("same_seed_repeat", t10, 42, "s42", rep)
            add("topk0_vs_neutral", k0, 42, "s42", 0)
            add("topk0_vs_neutral", t10, 42, "s42", 0)
    return jobs


def evaluate_controls(rows: list[dict]) -> list[dict]:
    """Turn control rows into pass/fail verdicts per model."""
    import pandas as pd
    df = pd.DataFrame([r for r in rows if not r.get("error")])
    res = []
    if df.empty:
        return res
    for (model, ctl), g in df.groupby(["model", "mode"]):
        n_pass = n = 0
        for _, gp in g.groupby("item_id"):
            n += 1
            outs = gp.sort_values(["config", "seed_mode", "rep"])
            if ctl == "greedy_repeat":
                n_pass += outs["output"].nunique() == 1
            elif ctl == "k1_vs_t0":
                n_pass += outs["output"].nunique() == 1
            elif ctl == "varies_t16":
                n_pass += outs["output"].nunique() >= 2
            elif ctl == "same_seed_repeat":
                n_pass += outs["output"].nunique() == 1
            elif ctl == "topk0_vs_neutral":
                n_pass += outs["output"].nunique() == 1
        expect = {
            "greedy_repeat": "PASS expected; FAIL => engine is not bit-deterministic at T=0 (report it!)",
            "k1_vs_t0": "PASS expected; FAIL => top_k=1 is not equivalent to T=0 in this build",
            "varies_t16": "PASS expected; FAIL on all prompts => options are not reaching the sampler",
            "same_seed_repeat": "PASS expected; FAIL => a fixed seed does not make sampling reproducible",
            "topk0_vs_neutral": "PASS => top_k=0 behaves as 'off' (same as 1000); FAIL => 0 means something else",
        }[ctl]
        res.append({"model": model, "control": ctl, "prompts": n, "passed": int(n_pass),
                    "rate": n_pass / n if n else float("nan"), "expectation": expect})
    return res


def count_jobs(jobs: list[dict]) -> Counter:
    return Counter((j["experiment"], j["model"]) for j in jobs)