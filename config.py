"""Central configuration: models, sampling configs, base options, prompt templates.

Design rules
------------
* Every Ollama option that can influence sampling is set EXPLICITLY on every call
  (no reliance on Ollama defaults, which differ between versions).
* Each sweep changes ONE knob and neutralises the others:
    - temperature sweep : top_k=NEUTRAL_TOP_K, top_p=1.0, min_p=0
    - top-k sweep       : T=1.0, top_p=1.0
    - top-p sweep       : T=1.0, top_k=NEUTRAL_TOP_K
    - interactions      : T=1.3 with top-p or top-k on, compared against plain T=1.3
* ``ollama_default`` reproduces Ollama's stock sampler (incl. repeat_penalty 1.1) as a reference.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("SB_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("SB_RESULTS_DIR", ROOT / "results"))
PLOTS_DIR = Path(os.environ.get("SB_PLOTS_DIR", ROOT / "plots"))

# ---- models (pin the quantisation you actually pulled and record it in the report) ----
MODELS_MAIN = ["qwen2.5:1.5b", "smollm2:1.7b"]
MODELS_SMOKE = ["qwen2.5:1.5b"]          # only to verify the pipeline, not for science
MODELS_SCALE = ["llama3.1:8b"]           # partial CPU offload on a 4 GB GPU: small subset only

# ---- options that are identical for every call ----
BASE_OPTIONS = {
    "min_p": 0.0,
    "repeat_penalty": 1.0,   # Ollama's default is 1.1; neutralised to remove a confound
    "num_ctx": 4096,         # fixed, otherwise Ollama picks a context size from free VRAM
    "num_predict": 160,      # hard cap; done_reason == "length" marks truncation
}
NEUTRAL_TOP_K = 1000         # "off". Control `topk0_vs_neutral` tells you if 0 also means off.
NEUTRAL_TOP_P = 1.0


def _c(name, t, k=NEUTRAL_TOP_K, p=NEUTRAL_TOP_P, group="", baseline="", **overrides):
    return {
        "name": name, "temperature": float(t), "top_k": int(k), "top_p": float(p),
        "group": group, "baseline": baseline, "overrides": overrides,
    }


CONFIG_LIST = [
    # temperature sweep (no truncation)
    _c("T0.0", 0.0, group="temperature"),
    _c("T0.3", 0.3, group="temperature", baseline="T0.0"),
    _c("T0.7", 0.7, group="temperature", baseline="T0.0"),
    _c("T1.0", 1.0, group="temperature", baseline="T0.0"),
    _c("T1.3", 1.3, group="temperature", baseline="T0.0"),
    _c("T1.6", 1.6, group="temperature", baseline="T0.0"),
    # top-k sweep at T=1.0 (baseline = untruncated T1.0)
    _c("T1.0_k1", 1.0, k=1, group="top_k", baseline="T1.0"),
    _c("T1.0_k10", 1.0, k=10, group="top_k", baseline="T1.0"),
    _c("T1.0_k40", 1.0, k=40, group="top_k", baseline="T1.0"),
    _c("T1.0_k100", 1.0, k=100, group="top_k", baseline="T1.0"),
    # top-p sweep at T=1.0
    _c("T1.0_p0.5", 1.0, p=0.5, group="top_p", baseline="T1.0"),
    _c("T1.0_p0.9", 1.0, p=0.9, group="top_p", baseline="T1.0"),
    _c("T1.0_p0.95", 1.0, p=0.95, group="top_p", baseline="T1.0"),
    # interactions: does truncation rescue a hot temperature?
    _c("T1.3_p0.9", 1.3, p=0.9, group="interaction", baseline="T1.3"),
    _c("T1.3_k40", 1.3, k=40, group="interaction", baseline="T1.3"),
    # reference: Ollama stock sampler
    _c("ollama_default", 0.8, k=40, p=0.9, group="reference", repeat_penalty=1.1),
]
CONFIGS = {c["name"]: c for c in CONFIG_LIST}

DETERMINISM_CONFIGS = ["T0.0", "T1.0_k1", "T0.7", "T1.0", "T1.3_p0.9"]
JSON_CONFIGS = ["T0.0", "T0.7", "T1.0", "T1.3", "T1.3_p0.9", "ollama_default"]
SMOKE_CONFIGS = ["T0.0", "T1.0", "T1.0_k1", "T1.6"]

# ---- prompt templates (template is a factor so prompt effects can be compared with sampling effects) ----
QA_INSTRUCTION = "Answer the question in one or two sentences (at most 40 words)."
TEMPLATES = {
    "plain": None,
    "careful": (
        "You are a careful assistant. If you are not sure, or the question rests on a "
        "false premise, say so instead of guessing."
    ),
}

DETERMINISM_SEEDS = {"fixed": 42}  # "varied" uses seed=rep, "unseeded" omits the seed
DETERMINISM_REPS = 10
FAKE_REPS = 3
JSON_REPS = 2

MAX_CONSECUTIVE_ERRORS = 5


def is_greedy(cfg: dict) -> bool:
    """Greedy decoding: repeats are redundant (and are tested separately as a control)."""
    return cfg["temperature"] <= 0 or cfg["top_k"] == 1


def options_for(cfg: dict, seed=None, num_predict=None) -> dict:
    """Full explicit options dict for one call."""
    opts = {**BASE_OPTIONS, **cfg.get("overrides", {})}
    opts["temperature"] = float(cfg["temperature"])
    opts["top_k"] = int(cfg["top_k"])
    opts["top_p"] = float(cfg["top_p"])
    if num_predict is not None:
        opts["num_predict"] = int(num_predict)
    if seed is not None:
        opts["seed"] = int(seed)
    return opts
