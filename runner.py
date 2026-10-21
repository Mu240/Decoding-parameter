"""Resumable experiment runner.

  python runner.py plan     --models llama3.2:3b qwen2.5:3b gemma3:4b
  python runner.py estimate --model llama3.2:3b          # 20-prompt timing run -> total hours
  python runner.py run --experiment all --models llama3.2:3b --n-questions 200
  python runner.py run --smoke --backend mock             # whole pipeline with no GPU
  python runner.py controls --models llama3.2:3b

Results go to results/<experiment>.jsonl, one append-only row per generation. Re-running the same
command skips completed keys, so a crash or Ctrl-C loses at most one generation.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import config as C
import experiments as E
from data_builders import determinism_prompts, fake_entity_items, json_tasks, load_truthfulqa
from io_utils import append_jsonl, done_keys, git_commit, read_jsonl, stable_int, utc_now, write_json

EXPERIMENTS = ["tqa_gen", "tqa_mc", "fake", "determinism", "json"]


class TooManyErrors(RuntimeError):
    pass


def make_backend(name: str, **kw):
    if name == "mock":
        from backend import MockBackend
        return MockBackend(**kw)
    from backend import OllamaBackend
    return OllamaBackend()


def execute(jobs: list[dict], backend, out_path: Path, run_id: str, max_errors=C.MAX_CONSECUTIVE_ERRORS,
            logprobs=False, verbose=True) -> dict:
    """Run jobs sequentially, appending a row per generation. Skips completed keys.
    Aborts after `max_errors` consecutive failures instead of logging thousands of error rows."""
    done = done_keys(out_path)
    todo = [j for j in jobs if j["key"] not in done]
    stats = {"total": len(jobs), "skipped": len(jobs) - len(todo), "ran": 0, "errors": 0}
    if verbose:
        print(f"[{out_path.name}] {len(jobs)} jobs, {stats['skipped']} already done, {len(todo)} to run")
    commit, consecutive, current_model = git_commit(), 0, None
    t_start = time.perf_counter()
    for n, j in enumerate(todo, 1):
        if j["model"] != current_model:
            current_model = j["model"]
            if verbose:
                print(f"  loading {current_model} (warm-up, excluded from timing)")
            backend.warmup(current_model)
        meta = backend.meta(j["model"])
        opts = C.options_for(j["cfg"], seed=j["seed"], num_predict=j["num_predict"])
        row = {"key": j["key"], "run_id": run_id, "timestamp": utc_now(), "git_commit": commit,
               "backend": "mock" if backend.is_mock else "ollama", "experiment": j["experiment"],
               "model": j["model"], "config": j["config"], "template": j["template"], "item_id": j["item_id"],
               "seed_mode": j["seed_mode"], "rep": j["rep"], "mode": j["mode"], "prompt": j["prompt"],
               "system": j["system"], "options": opts, "meta": j["meta"], **meta}
        try:
            r = backend.generate(j["model"], j["prompt"], opts, system=j["system"], fmt=j["fmt"],
                                 mock_hint=j["hint"], logprobs=logprobs)
            row.update(r)
            consecutive = 0
            stats["ran"] += 1
        except Exception as e:  # noqa: BLE001 - we want to log whatever the client raised
            row["error"] = f"{type(e).__name__}: {e}"
            stats["errors"] += 1
            consecutive += 1
        append_jsonl(out_path, row)
        if consecutive >= max_errors:
            raise TooManyErrors(f"{consecutive} consecutive errors, last: {row.get('error')}. "
                                f"Is Ollama running and the model pulled? Re-run to resume.")
        if verbose and n % 200 == 0:
            el = time.perf_counter() - t_start
            print(f"  {n}/{len(todo)}  {el/n:.2f}s/gen  ETA {(len(todo)-n)*el/n/3600:.2f} h")
    return stats


def build_jobs(experiment, models, args, data_dir):
    cfgs_all = list(args.configs) if args.configs else list(C.CONFIGS)
    if args.smoke:
        cfgs_all = C.SMOKE_CONFIGS
    templates = tuple(args.templates)
    if experiment in ("tqa_gen", "tqa_mc"):
        qs = load_truthfulqa(Path(data_dir) / "truthfulqa_sample.jsonl", args.n_questions)
        if experiment == "tqa_gen":
            return E.plan_tqa_gen(models, cfgs_all, qs, templates)
        return E.plan_tqa_mc(models, cfgs_all, qs)
    if experiment == "fake":
        items = fake_entity_items()
        if args.smoke:
            items = items[:3] + items[-2:]
        return E.plan_fake(models, cfgs_all, items, reps=2 if args.smoke else C.FAKE_REPS, templates=templates)
    if experiment == "determinism":
        cfgs = [c for c in C.DETERMINISM_CONFIGS if c in cfgs_all] or C.SMOKE_CONFIGS[:2]
        prompts = determinism_prompts()
        if args.smoke:
            prompts = prompts[::6]
        return E.plan_determinism(models, cfgs, prompts, reps=3 if args.smoke else args.det_reps,
                                  seed_modes=tuple(args.det_seed_modes), num_predict=args.det_num_predict)
    if experiment == "json":
        cfgs = [c for c in C.JSON_CONFIGS if c in cfgs_all] or C.SMOKE_CONFIGS[:2]
        tasks = json_tasks(10 if args.smoke else args.n_json)
        return E.plan_json(models, cfgs, tasks)
    raise ValueError(experiment)


def cmd_plan(args):
    total = 0
    for exp in EXPERIMENTS:
        jobs = build_jobs(exp, args.models, args, args.data_dir)
        for (e, m), n in sorted(E.count_jobs(jobs).items()):
            print(f"{e:12s} {m:16s} {n:7d} generations")
            total += n
    print(f"TOTAL {total} generations (before the controls). Run `estimate` to turn this into hours.")


def cmd_estimate(args):
    """Time N real generations, then project total hours = jobs x avg_tokens / tokens-per-second."""
    be = make_backend(args.backend)
    qs = load_truthfulqa(Path(args.data_dir) / "truthfulqa_sample.jsonl", args.timing_n)
    jobs = E.plan_tqa_gen([args.model], ["T0.7"], qs)
    be.warmup(args.model)
    toks, secs = 0, 0.0
    for j in jobs:
        r = be.generate(args.model, j["prompt"], C.options_for(j["cfg"], seed=j["seed"]))
        toks += r.get("eval_count") or 0
        secs += (r.get("eval_duration") or 0) / 1e9 or r.get("wall_s", 0.0)
    tps = toks / secs if secs else float("nan")
    avg_tok = toks / len(jobs)
    all_jobs = sum((build_jobs(e, [args.model], args, args.data_dir) for e in EXPERIMENTS), [])
    hours = len(all_jobs) * avg_tok / tps / 3600 if tps == tps and tps > 0 else float("nan")
    print(f"{args.model}: {tps:.1f} tokens/s, {avg_tok:.1f} tokens/answer (QA).")
    print(f"Planned {len(all_jobs)} generations -> about {hours:.1f} h for this model "
          f"(JSON answers are longer, so treat this as a lower bound).")


def cmd_run(args):
    results = Path(args.results_dir)
    run_id = args.run_id or time.strftime("run-%Y%m%d-%H%M%S")
    models = C.MODELS_SMOKE if args.smoke and not args.models else args.models
    be = make_backend(args.backend)
    exps = EXPERIMENTS if args.experiment == "all" else [args.experiment]
    write_json(results / f"{run_id}_manifest.json", {
        "run_id": run_id, "models": models, "experiments": exps, "git_commit": git_commit(),
        "ollama_version": be.version(), "base_options": C.BASE_OPTIONS, "smoke": args.smoke,
        "configs": {k: {x: v[x] for x in ("temperature", "top_k", "top_p", "overrides")} for k, v in C.CONFIGS.items()},
        "backend": args.backend, "started": utc_now()})
    for exp in exps:
        jobs = build_jobs(exp, models, args, args.data_dir)
        print(execute(jobs, be, results / f"{exp}.jsonl", run_id, logprobs=args.logprobs))
    if args.controls or args.experiment == "all":
        cmd_controls(args, be=be, models=models)


def cmd_controls(args, be=None, models=None):
    results = Path(args.results_dir)
    be = be or make_backend(args.backend)
    models = models or args.models
    jobs = E.plan_controls(models, determinism_prompts())
    execute(jobs, be, results / "controls.jsonl", args.run_id or "controls")
    verdicts = E.evaluate_controls(read_jsonl(results / "controls.jsonl"))
    write_json(results / "controls_report.json", verdicts)
    for v in verdicts:
        print(f"[{v['model']}] {v['control']:18s} {v['passed']}/{v['prompts']}  {v['expectation']}")


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "estimate", "run", "controls"])
    ap.add_argument("--models", nargs="+", default=C.MODELS_MAIN)
    ap.add_argument("--model", default=C.MODELS_MAIN[0], help="single model for `estimate`")
    ap.add_argument("--experiment", default="all", choices=["all"] + EXPERIMENTS)
    ap.add_argument("--n-questions", type=int, default=200)
    ap.add_argument("--n-json", type=int, default=60)
    ap.add_argument("--timing-n", type=int, default=20)
    ap.add_argument("--configs", nargs="+", default=None, choices=list(C.CONFIGS),
                    help="subset of sampling configs (use for tqa_gen / tqa_mc / fake; keep each baseline in the set)")
    ap.add_argument("--det-reps", type=int, default=C.DETERMINISM_REPS, help="repeats per prompt (determinism)")
    ap.add_argument("--det-seed-modes", nargs="+", default=["fixed", "varied", "unseeded"],
                    choices=["fixed", "varied", "unseeded"], help="seed regimes to run (determinism)")
    ap.add_argument("--det-num-predict", type=int, default=None,
                    help="cap output tokens for determinism (changing it makes old rows inconsistent: start a fresh file)")
    ap.add_argument("--templates", nargs="+", default=["plain"], choices=list(C.TEMPLATES))
    ap.add_argument("--backend", default="ollama", choices=["ollama", "mock"])
    ap.add_argument("--smoke", action="store_true", help="tiny run on the 1.5B model to verify the pipeline")
    ap.add_argument("--controls", action="store_true", help="also run the sampler sanity controls")
    ap.add_argument("--logprobs", action="store_true", help="request per-token logprobs (needs recent Ollama)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--results-dir", default=str(C.RESULTS_DIR))
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.smoke:
        args.n_questions = min(args.n_questions, 10)
    try:
        {"plan": cmd_plan, "estimate": cmd_estimate, "run": cmd_run, "controls": cmd_controls}[args.cmd](args)
    except TooManyErrors as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()