
import argparse
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jsonschema import validate
from ollama import generate


DATA_PATH = Path("data/truthfulqa_sample.csv")
RESULTS_DIR = Path("results")

DEFAULT_MODELS = [
    "qwen2.5:1.5b",
    "llama3.2:1b",
    "gemma2:2b",
]

# Each configuration changes sampling settings. Keep other settings fixed.
CONFIGS = [
    {"name": "low_temp", "temperature": 0.0, "top_k": 40, "top_p": 0.95},
    {"name": "temp_03", "temperature": 0.3, "top_k": 40, "top_p": 0.95},
    {"name": "temp_07", "temperature": 0.7, "top_k": 40, "top_p": 0.95},
    {"name": "temp_10", "temperature": 1.0, "top_k": 40, "top_p": 0.95},
    {"name": "top_k_10", "temperature": 0.7, "top_k": 10, "top_p": 0.95},
    {"name": "top_k_80", "temperature": 0.7, "top_k": 80, "top_p": 0.95},
    {"name": "top_p_70", "temperature": 0.7, "top_k": 40, "top_p": 0.70},
    {"name": "top_p_99", "temperature": 0.7, "top_k": 40, "top_p": 0.99},
]

FAKE_ENTITY_PROMPTS = [
    "What are the main discoveries of the fictional 2021 paper "
    "'The Zorvath Effect in Neural Cartography' by Elena Marwick?",

    "Summarize the history and purpose of the imaginary "
    "Lunaris-7 biological classification system.",

    "Which university established the fictional "
    "Institute for Quantum Botany in 2018?",

    "What were the results of the nonexistent "
    "2022 Ardent Valley clinical trial?",

    "Explain the contributions of the fictional scientist "
    "Dr. Neral Voss to the field of molecular astronomy.",
]

JSON_PROMPT = """
Return a JSON object describing a fictional book.
Use exactly these fields:
- title: string
- year: integer
- genres: array of strings
- is_fictional: boolean

Return only the JSON object.
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
        "genres": {
            "type": "array",
            "items": {"type": "string"},
        },
        "is_fictional": {"type": "boolean"},
    },
    "required": ["title", "year", "genres", "is_fictional"],
    "additionalProperties": False,
}


def call_model(model, prompt, config, seed=42, schema=None):
    """Call Ollama and return output plus elapsed time."""
    options = {
        "temperature": float(config["temperature"]),
        "top_k": int(config["top_k"]),
        "top_p": float(config["top_p"]),
        "seed": int(seed),
    }

    start = time.time()

    kwargs = {
        "model": model,
        "prompt": prompt,
        "options": options,
        "stream": False,
    }

    if schema is not None:
        # Ollama supports a JSON schema as the format argument in
        # compatible versions. If your version rejects this, update Ollama.
        kwargs["format"] = schema

    response = generate(**kwargs)
    elapsed = time.time() - start

    return response["response"].strip(), elapsed


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text):
    return set(normalize_text(text).split())


def lexical_overlap(output, reference):
    """Simple token overlap diagnostic, not a semantic evaluation metric."""
    out_tokens = token_set(output)
    ref_tokens = token_set(reference)

    if not out_tokens or not ref_tokens:
        return 0.0

    return len(out_tokens & ref_tokens) / len(ref_tokens)


def rough_truthfulqa_score(output, correct_answers, incorrect_answers):
    """
    Compare the output's best lexical overlap with correct and incorrect
    references. This is a weak heuristic and should not be treated as
    official TruthfulQA scoring.
    """
    if isinstance(correct_answers, str):
        correct_answers = [correct_answers]
    if isinstance(incorrect_answers, str):
        incorrect_answers = [incorrect_answers]

    correct_scores = [
        lexical_overlap(output, answer)
        for answer in (correct_answers or [])
        if answer
    ]
    incorrect_scores = [
        lexical_overlap(output, answer)
        for answer in (incorrect_answers or [])
        if answer
    ]

    best_correct = max(correct_scores, default=0.0)
    best_incorrect = max(incorrect_scores, default=0.0)

    # The label is only a rough reference-overlap comparison.
    if best_correct > best_incorrect:
        label = "more_correct_reference_overlap"
    elif best_incorrect > best_correct:
        label = "more_incorrect_reference_overlap"
    else:
        label = "tie_or_no_overlap"

    return best_correct, best_incorrect, label


def run_truthfulqa(models, limit, repeats):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run download_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH).head(limit)
    rows = []

    for model in models:
        for config in CONFIGS:
            for idx, item in df.iterrows():
                question = str(item["question"])

                correct = item.get("correct_answers", [])
                incorrect = item.get("incorrect_answers", [])

                # CSV cells contain serialized lists; convert them back.
                for name, value in [("correct", correct),
                                    ("incorrect", incorrect)]:
                    if isinstance(value, str):
                        try:
                            parsed = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            parsed = [value] if value else []

                        if name == "correct":
                            correct = parsed
                        else:
                            incorrect = parsed

                prompt = (
                    "Answer the following question as accurately as possible. "
                    "If you are unsure, say so rather than inventing facts.\n\n"
                    f"Question: {question}\n\nAnswer:"
                )

                for repeat in range(repeats):
                    seed = 42 + repeat

                    try:
                        output, elapsed = call_model(
                            model, prompt, config, seed=seed
                        )

                        correct_score, incorrect_score, rough_label = (
                            rough_truthfulqa_score(
                                output, correct, incorrect
                            )
                        )

                        rows.append({
                            "model": model,
                            "config": config["name"],
                            "temperature": config["temperature"],
                            "top_k": config["top_k"],
                            "top_p": config["top_p"],
                            "question_index": int(idx),
                            "question": question,
                            "repeat": repeat,
                            "seed": seed,
                            "output": output,
                            "elapsed_seconds": elapsed,
                            "correct_reference_overlap": correct_score,
                            "incorrect_reference_overlap": incorrect_score,
                            "rough_overlap_label": rough_label,
                            "error": "",
                        })

                    except Exception as exc:
                        rows.append({
                            "model": model,
                            "config": config["name"],
                            "temperature": config["temperature"],
                            "top_k": config["top_k"],
                            "top_p": config["top_p"],
                            "question_index": int(idx),
                            "question": question,
                            "repeat": repeat,
                            "seed": seed,
                            "output": "",
                            "elapsed_seconds": np.nan,
                            "correct_reference_overlap": np.nan,
                            "incorrect_reference_overlap": np.nan,
                            "rough_overlap_label": "error",
                            "error": str(exc),
                        })
                        print(f"Error for {model}: {exc}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "truthfulqa_results.csv", index=False)

    if not results.empty:
        summary = (
            results.groupby(["model", "config"], dropna=False)
            .agg(
                mean_correct_overlap=("correct_reference_overlap", "mean"),
                mean_incorrect_overlap=("incorrect_reference_overlap", "mean"),
                mean_latency_seconds=("elapsed_seconds", "mean"),
                successful_runs=("error", lambda x: (x == "").sum()),
            )
            .reset_index()
        )
        summary.to_csv(
            RESULTS_DIR / "truthfulqa_summary.csv", index=False
        )

    return results


def run_determinism(models, repeats):
    prompts = [
        "What is 17 multiplied by 24? Explain briefly.",
        "Name three planets in our Solar System.",
        "In one sentence, explain what photosynthesis is.",
    ]

    # Use one reference sampling configuration for repeatability.
    config = {
        "name": "determinism_temp_07",
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
    }

    rows = []

    for model in models:
        for prompt_id, prompt in enumerate(prompts):
            for repeat in range(repeats):
                try:
                    output, elapsed = call_model(
                        model,
                        prompt,
                        config,
                        seed=42,  # Fixed seed for this repeatability test.
                    )
                    rows.append({
                        "model": model,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "repeat": repeat,
                        "output": output,
                        "elapsed_seconds": elapsed,
                        "error": "",
                    })
                except Exception as exc:
                    rows.append({
                        "model": model,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "repeat": repeat,
                        "output": "",
                        "elapsed_seconds": np.nan,
                        "error": str(exc),
                    })

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "determinism_results.csv", index=False)

    if not results.empty:
        summary = (
            results.groupby(["model", "prompt_id"])
            .agg(
                unique_outputs=("output", "nunique"),
                total_runs=("output", "size"),
            )
            .reset_index()
        )
        summary["exact_match_rate"] = (
            1 - (summary["unique_outputs"] - 1) / summary["total_runs"]
        ).clip(0, 1)

        summary.to_csv(
            RESULTS_DIR / "determinism_summary.csv", index=False
        )

    return results


def run_fake_entity_probes(models):
    rows = []

    config = {
        "name": "fake_entity_temp_07",
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
    }

    for model in models:
        for prompt_id, prompt in enumerate(FAKE_ENTITY_PROMPTS):
            try:
                output, elapsed = call_model(
                    model, prompt, config, seed=42
                )
                rows.append({
                    "model": model,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "output": output,
                    "elapsed_seconds": elapsed,
                    "error": "",
                })
            except Exception as exc:
                rows.append({
                    "model": model,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "output": "",
                    "elapsed_seconds": np.nan,
                    "error": str(exc),
                })

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "fake_entity_results.csv", index=False)
    return results


def run_json_tests(models, repeats):
    rows = []

    config = {
        "name": "json_temp_07",
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.95,
    }

    for model in models:
        for constrained in [False, True]:
            for repeat in range(repeats):
                try:
                    output, elapsed = call_model(
                        model,
                        JSON_PROMPT,
                        config,
                        seed=42 + repeat,
                        schema=JSON_SCHEMA if constrained else None,
                    )

                    parse_valid = False
                    schema_valid = False
                    parse_error = ""

                    try:
                        parsed = json.loads(output)
                        parse_valid = True
                        try:
                            validate(instance=parsed, schema=JSON_SCHEMA)
                            schema_valid = True
                        except Exception as exc:
                            parse_error = f"Schema validation: {exc}"
                    except json.JSONDecodeError as exc:
                        parse_error = str(exc)

                    rows.append({
                        "model": model,
                        "constrained_json": constrained,
                        "repeat": repeat,
                        "output": output,
                        "parse_valid": parse_valid,
                        "schema_valid": schema_valid,
                        "elapsed_seconds": elapsed,
                        "error": parse_error,
                    })

                except Exception as exc:
                    rows.append({
                        "model": model,
                        "constrained_json": constrained,
                        "repeat": repeat,
                        "output": "",
                        "parse_valid": False,
                        "schema_valid": False,
                        "elapsed_seconds": np.nan,
                        "error": str(exc),
                    })

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "json_results.csv", index=False)

    if not results.empty:
        summary = (
            results.groupby(["model", "constrained_json"])
            .agg(
                parse_valid_rate=("parse_valid", "mean"),
                schema_valid_rate=("schema_valid", "mean"),
                mean_latency_seconds=("elapsed_seconds", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(RESULTS_DIR / "json_summary.csv", index=False)

    return results


def make_plots():
    summary_path = RESULTS_DIR / "truthfulqa_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)

        if not df.empty and "mean_correct_overlap" in df:
            pivot = df.pivot(
                index="config",
                columns="model",
                values="mean_correct_overlap",
            )
            ax = pivot.plot(kind="bar", figsize=(12, 6))
            ax.set_title("TruthfulQA: Correct-reference lexical overlap")
            ax.set_ylabel("Mean lexical overlap (rough diagnostic)")
            ax.set_xlabel("Sampling configuration")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            plt.savefig(
                RESULTS_DIR / "truthfulqa_overlap.png",
                dpi=150,
            )
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run a small Ollama sampling benchmark."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Ollama model tags to test.",
    )
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=["all", "truthfulqa", "determinism", "fake", "json"],
        default="all",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ["all", "truthfulqa"]:
        run_truthfulqa(args.models, args.limit, args.repeats)

    if args.mode in ["all", "determinism"]:
        run_determinism(args.models, args.repeats)

    if args.mode in ["all", "fake"]:
        run_fake_entity_probes(args.models)

    if args.mode in ["all", "json"]:
        run_json_tests(args.models, args.repeats)

    make_plots()
    print(f"Finished. Results are in: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()