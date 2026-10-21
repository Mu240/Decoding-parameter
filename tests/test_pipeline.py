import json
from pathlib import Path

import pytest

import config as C
import experiments as E
from analysis import analyze
from backend import MockBackend
from data_builders import (determinism_prompts, fake_entity_items, json_tasks, load_truthfulqa, make_tqa_fixture,
                           verification_checklist)
from download_datasets import build_rows, stratified_sample
from io_utils import read_jsonl
from runner import TooManyErrors, execute, main as runner_main


@pytest.fixture()
def dirs(tmp_path):
    d = {"data": tmp_path / "data", "results": tmp_path / "results", "plots": tmp_path / "plots"}
    d["data"].mkdir()
    make_tqa_fixture(d["data"] / "truthfulqa_sample.jsonl", n=12)
    return d


# ---- configuration hygiene ----
def test_every_config_sets_all_sampler_options_explicitly():
    for cfg in C.CONFIGS.values():
        o = C.options_for(cfg, seed=1)
        for k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty", "num_ctx", "num_predict", "seed"):
            assert k in o, (cfg["name"], k)


def test_sweeps_isolate_one_knob():
    for name, cfg in C.CONFIGS.items():
        if cfg["group"] == "temperature":
            assert cfg["top_k"] == C.NEUTRAL_TOP_K and cfg["top_p"] == 1.0
        if cfg["group"] == "top_k":
            assert cfg["temperature"] == 1.0 and cfg["top_p"] == 1.0
        if cfg["group"] == "top_p":
            assert cfg["temperature"] == 1.0 and cfg["top_k"] == C.NEUTRAL_TOP_K
        if cfg["baseline"]:
            assert cfg["baseline"] in C.CONFIGS


def test_config_count_and_reference():
    assert len(C.CONFIGS) == 16  # 15 screening configs + Ollama default
    assert C.CONFIGS["ollama_default"]["overrides"]["repeat_penalty"] == 1.1


def test_unseeded_options_omit_seed():
    assert "seed" not in C.options_for(C.CONFIGS["T1.0"], seed=None)


# ---- item banks ----
def test_banks_shapes():
    assert len(determinism_prompts()) == 30
    items = fake_entity_items()
    assert sum(i["category"] != "real_control" for i in items) == 45
    assert not any(w in i["prompt"].lower() for i in items for w in ("fictional", "imaginary", "nonexistent"))
    assert "fake-person-0" in verification_checklist()
    t = json_tasks(60)
    assert len(t) == 60 and json_tasks(60) == t  # deterministic


def test_json_gold_is_schema_valid_and_in_prompt_text():
    from scorers import schema_errors
    for t in json_tasks(30):
        assert schema_errors(t["gold"], t["schema"]) == []


def test_stratified_sample_and_list_fields_survive():
    gen = [{"question": f"q{i}", "category": "A" if i % 2 else "B", "best_answer": "b",
            "correct_answers": ["x", "y"], "incorrect_answers": ["z"]} for i in range(20)]
    mc = [{"question": f"q{i}", "mc1_targets": {"choices": ["x", "z"], "labels": [1, 0]}} for i in range(20)]
    rows = build_rows(gen, mc)
    s = stratified_sample(rows, 8, seed=0)
    assert len(s) == 8 and all(isinstance(r["correct_answers"], list) and len(r["correct_answers"]) == 2 for r in s)
    assert s == stratified_sample(rows, 8, seed=0)


# ---- planning ----
def test_keys_unique_and_greedy_not_repeated():
    qs = load_truthfulqa(Path(C.DATA_DIR) / "x") if False else None  # noqa
    jobs = E.plan_fake(["m"], ["T0.0", "T1.0"], fake_entity_items()[:4], reps=3)
    assert len({j["key"] for j in jobs}) == len(jobs)
    assert sum(j["config"] == "T0.0" for j in jobs) == 4      # greedy: 1 rep
    assert sum(j["config"] == "T1.0" for j in jobs) == 12


def test_determinism_plan_has_three_seed_regimes():
    jobs = E.plan_determinism(["m"], ["T1.0"], determinism_prompts()[:1], reps=4)
    by = {}
    for j in jobs:
        by.setdefault(j["seed_mode"], []).append(j["seed"])
    assert by["fixed"] == [42] * 4 and by["varied"] == [0, 1, 2, 3] and by["unseeded"] == [None] * 4


def test_mc_layout_is_deterministic_and_points_at_correct_option(dirs):
    qs = load_truthfulqa(dirs["data"] / "truthfulqa_sample.jsonl")
    for q in qs:
        order, pos = E.mc_layout(q)
        assert q["mc1_choices"][order[pos]] == q["best_answer"]
        assert (order, pos) == E.mc_layout(q)


# ---- runner: resume + failure handling ----
def test_resume_skips_completed_and_never_duplicates(dirs):
    jobs = E.plan_fake(["m"], ["T1.0"], fake_entity_items()[:5], reps=2)
    out = dirs["results"] / "fake.jsonl"
    be = MockBackend()
    s1 = execute(jobs[:4], be, out, "r1", verbose=False)
    s2 = execute(jobs, be, out, "r2", verbose=False)
    assert s1["ran"] == 4 and s2["skipped"] == 4 and s2["ran"] == len(jobs) - 4
    keys = [r["key"] for r in read_jsonl(out)]
    assert len(keys) == len(set(keys)) == len(jobs)


def test_rows_record_provenance_and_options(dirs):
    jobs = E.plan_fake(["m"], ["T1.0"], fake_entity_items()[:1], reps=1)
    execute(jobs, MockBackend(), dirs["results"] / "f.jsonl", "r", verbose=False)
    row = read_jsonl(dirs["results"] / "f.jsonl")[0]
    for k in ("options", "model_digest", "ollama_version", "done_reason", "eval_count", "git_commit", "run_id",
              "timestamp", "prompt", "output", "seed_mode"):
        assert k in row
    assert row["options"]["top_k"] == C.NEUTRAL_TOP_K and row["options"]["num_ctx"] == 4096


def test_aborts_after_consecutive_errors_and_retries_on_resume(dirs):
    jobs = E.plan_fake(["m"], ["T1.0"], fake_entity_items()[:10], reps=1)
    out = dirs["results"] / "f.jsonl"
    with pytest.raises(TooManyErrors):
        execute(jobs, MockBackend(fail_on=lambda p: True), out, "r", max_errors=3, verbose=False)
    assert len(read_jsonl(out)) == 3                         # not thousands of error rows
    stats = execute(jobs, MockBackend(), out, "r2", verbose=False)
    assert stats["ran"] == len(jobs)                         # error rows are retried, not treated as done


# ---- controls ----
def test_controls_pass_on_ideal_mock_and_flag_engine_flake(dirs):
    jobs = E.plan_controls(["m"], determinism_prompts())
    execute(jobs, MockBackend(), dirs["results"] / "c.jsonl", "r", verbose=False)
    v = {x["control"]: x for x in E.evaluate_controls(read_jsonl(dirs["results"] / "c.jsonl"))}
    assert all(v[c]["rate"] == 1.0 for c in ("greedy_repeat", "k1_vs_t0", "varies_t16", "same_seed_repeat"))
    flaky = dirs["results"] / "c2.jsonl"
    execute(jobs, MockBackend(flake_rate=0.6), flaky, "r", verbose=False)
    v2 = {x["control"]: x for x in E.evaluate_controls(read_jsonl(flaky))}
    assert v2["greedy_repeat"]["rate"] < 1.0                 # the control catches nondeterminism


# ---- mock sanity (so end-to-end plots are meaningful) ----
def test_mock_error_rate_grows_with_temperature():
    be = MockBackend()
    hint = {"kind": "json", "ideal": {"a": "x", "n": 3}}
    from scorers import score_json
    schema = json_tasks(1)[0]["schema"]
    rates = []
    for T in (0.0, 1.6):
        ok = 0
        for s in range(200):
            o = be.generate("m", "p", C.options_for({"temperature": T, "top_k": 1000, "top_p": 1.0, "overrides": {}}, seed=s),
                            mock_hint=hint)
            ok += score_json(o["output"], schema, {"a": "x", "n": 3}, o["done_reason"])["strict_parse"]
        rates.append(ok / 200)
    assert rates[0] == 1.0 and rates[1] < 0.9


# ---- full pipeline on the mock backend ----
def test_end_to_end_smoke_run_and_analysis(dirs):
    argv = ["run", "--backend", "mock", "--smoke", "--experiment", "all", "--models", "mock-model",
            "--data-dir", str(dirs["data"]), "--results-dir", str(dirs["results"]), "--n-questions", "10"]
    runner_main(argv)
    runner_main(argv)                                        # second invocation must be a pure no-op (resume)
    produced = analyze(dirs["results"], dirs["data"], dirs["results"] / "tables", dirs["plots"])
    assert set(produced) == {"tqa_gen", "tqa_mc", "fake", "determinism", "json"}
    tables = {p.name for p in (dirs["results"] / "tables").iterdir()}
    for t in ("tqa_gen_summary.csv", "tqa_gen_contrasts.csv", "tqa_gen_gee.csv", "determinism_summary.csv",
              "json_summary.csv", "json_failure_taxonomy.csv", "fake_summary.csv", "tqa_mc_summary.csv"):
        assert t in tables
    assert (dirs["plots"] / "determinism_by_config.png").exists()
    assert (dirs["plots"] / "json_validity.png").exists()
    report = json.loads((dirs["results"] / "controls_report.json").read_text())
    assert report and all(r["rate"] == 1.0 for r in report)
