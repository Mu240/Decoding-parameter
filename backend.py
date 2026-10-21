"""Generation backends: a real Ollama client and a mock for dry runs / tests.

Both expose:  meta(model), warmup(model), generate(model, prompt, options, ...)
and ``generate`` returns the same dict shape.

The mock is NOT a model of real LLM behaviour. It only lets the whole pipeline
(planning, logging, resume, scoring, analysis, plots) be exercised without a GPU.
Never report numbers produced by the mock.
"""
from __future__ import annotations

import json
import os
import random
import re
import time

from config import BASE_OPTIONS
from io_utils import stable_int


def _plain(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if isinstance(o, (list, tuple)):
        return [_plain(x) for x in o]
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    return o


def _get(o, key, default=None):
    if o is None:
        return default
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


class OllamaBackend:
    is_mock = False

    def __init__(self, host=None, timeout=300, keep_alive="30m"):
        from ollama import Client  # imported lazily so the mock needs no install

        host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        if "://" not in host:
            host = "http://" + host
        self.host = host
        self.keep_alive = keep_alive
        self.client = Client(host=host, timeout=timeout)
        self._meta_cache: dict[str, dict] = {}
        self._version = None

    # ---- provenance ----
    def version(self) -> str:
        if self._version is None:
            try:
                import httpx
                self._version = httpx.get(self.host + "/api/version", timeout=10).json().get("version", "unknown")
            except Exception:
                self._version = "unknown"
        return self._version

    def meta(self, model: str) -> dict:
        if model in self._meta_cache:
            return self._meta_cache[model]
        info = {"ollama_version": self.version(), "model_digest": "unknown", "quantization": "unknown",
                "parameter_size": "unknown"}
        try:
            for m in _get(self.client.list(), "models", []):
                name = _get(m, "model") or _get(m, "name")
                if name == model or name == model + ":latest":
                    info["model_digest"] = _get(m, "digest", "unknown")
        except Exception:
            pass
        try:
            det = _get(self.client.show(model), "details")
            info["quantization"] = _get(det, "quantization_level", "unknown")
            info["parameter_size"] = _get(det, "parameter_size", "unknown")
        except Exception:
            pass
        self._meta_cache[model] = info
        return info

    def warmup(self, model: str) -> None:
        """Load the model so load time never lands in a measured generation."""
        self.client.generate(model=model, prompt="Hi", keep_alive=self.keep_alive,
                             options={**BASE_OPTIONS, "temperature": 0.0, "num_predict": 1})

    # ---- generation ----
    def generate(self, model, prompt, options, system=None, fmt=None, mock_hint=None, logprobs=False):
        kwargs = {}
        if system:
            kwargs["system"] = system
        if fmt is not None:
            kwargs["format"] = fmt
        if logprobs:  # needs a recent Ollama + client; verify on your install
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 5
        t0 = time.perf_counter()
        r = self.client.generate(model=model, prompt=prompt, options=options, stream=False,
                                 keep_alive=self.keep_alive, **kwargs)
        wall = time.perf_counter() - t0
        out = {
            "output": (_get(r, "response", "") or "").strip(),
            "wall_s": wall,
            "done_reason": _get(r, "done_reason"),
            "eval_count": _get(r, "eval_count"),
            "prompt_eval_count": _get(r, "prompt_eval_count"),
            "eval_duration": _get(r, "eval_duration"),
            "load_duration": _get(r, "load_duration"),
            "total_duration": _get(r, "total_duration"),
        }
        if logprobs:
            out["logprobs"] = _plain(_get(r, "logprobs"))
        return out


# ----------------------------------------------------------------------------------
class MockBackend:
    """Deterministic fake LLM whose error rate grows with the *effective* temperature."""

    is_mock = True
    WRAPPERS = ["", "", "Well, ", "Generally, ", "In short, ", "Briefly: "]

    def __init__(self, flake_rate=0.0, fail_on=None):
        self.flake_rate = flake_rate     # chance a *greedy* call differs (emulates engine nondeterminism)
        self.fail_on = fail_on           # callable(prompt)->bool, raises to test error handling
        self._ctr = 0
        self.calls = 0

    def version(self):
        return "mock"

    def meta(self, model):
        return {"ollama_version": "mock", "model_digest": "mock-" + model, "quantization": "mock",
                "parameter_size": "mock"}

    def warmup(self, model):
        return None

    @staticmethod
    def effective_temperature(o):
        t = float(o["temperature"])
        if o.get("top_p", 1.0) < 1.0:
            t *= 0.6 if o["top_p"] <= 0.9 else 0.8
        k = o.get("top_k", 1000)
        if 0 < k <= 40:              # the mock treats top_k=0 as "off", like llama.cpp
            t *= 0.7
        return t

    def generate(self, model, prompt, options, system=None, fmt=None, mock_hint=None, logprobs=False):
        self.calls += 1
        if self.fail_on is not None and self.fail_on(prompt):
            raise ConnectionError("mock backend failure")
        greedy = options["temperature"] <= 0 or options.get("top_k", 1000) == 1
        seed = options.get("seed")
        if seed is None:
            self._ctr += 1
            seed = f"auto{self._ctr}"
        k_eff = options.get("top_k") or 1000   # 0 and 1000 both mean "no top-k" in the mock
        rng = random.Random(stable_int(f"{model}|{prompt}|{system}|{seed}|{options['temperature']}|"
                                       f"{k_eff}|{options.get('top_p')}"))
        err = 0.0 if greedy else min(0.9, max(0.0, (self.effective_temperature(options) - 0.6) / 1.4))
        hint = mock_hint or {"kind": "free"}
        done = "stop"
        kind = hint.get("kind", "free")

        if kind == "qa":
            wrap = "" if greedy else rng.choice(self.WRAPPERS)
            if rng.random() < err * 0.6:
                text = wrap + hint.get("wrong", "I think the common belief is true.")
            else:
                text = wrap + hint.get("ideal", "Unknown.")
        elif kind == "mc":
            n = hint["n_choices"]
            ideal = hint["ideal_idx"]
            idx = ideal if rng.random() >= err * 0.8 else rng.randrange(n)
            text = chr(65 + idx)
        elif kind == "fake":
            p_fab = 0.15 + 0.7 * err
            if greedy:
                p_fab = 0.3
                rng = random.Random(stable_int(prompt))  # greedy picks one fixed behaviour per prompt
            if rng.random() < p_fab:
                text = "It is best known for pioneering work in the field, published in 2004, with major results on scaling."
            else:
                text = "I'm not aware of any record of this; it may not exist, so I can't verify details."
        elif kind == "real":
            text = ("I'm not sure I can verify that." if rng.random() < err * 0.3
                    else "It is a well-documented topic with a clear and widely known answer.")
        elif kind == "json":
            gold = hint["ideal"]
            obj = dict(gold)
            if rng.random() < err * 0.5:          # content error
                k = rng.choice(list(obj))
                obj[k] = "WRONG" if isinstance(obj[k], str) else (obj[k] or 0) + 1
            raw = json.dumps(obj)
            if fmt is None and rng.random() < err * 0.8:   # structural error (prompt-only mode)
                mode = rng.choice(["fence", "prose", "comma", "trunc", "missing"])
                if mode == "fence":
                    raw = "```json\n" + raw + "\n```"
                elif mode == "prose":
                    raw = "Sure! Here is the JSON:\n" + raw + "\nHope that helps."
                elif mode == "comma":
                    raw = raw[:-1] + ",}"
                elif mode == "trunc":
                    raw, done = raw[: max(2, len(raw) // 2)], "length"
                else:
                    obj.pop(next(iter(obj)))
                    raw = json.dumps(obj)
            text = raw
        else:  # free text
            vocab = ["alpha", "river", "stone", "quick", "lantern", "story", "echo", "north", "amber", "silent"]
            if greedy:
                r2 = random.Random(stable_int(prompt))
                text = " ".join(r2.choice(vocab) for _ in range(12))
            else:
                text = " ".join(rng.choice(vocab) for _ in range(12))
        if greedy and self.flake_rate and rng.random() < self.flake_rate:
            text += " (alt)"
        out = {"output": text, "wall_s": 0.0, "done_reason": done, "eval_count": len(text.split()),
               "prompt_eval_count": len(prompt.split()), "eval_duration": 1_000_000 * max(1, len(text.split())),
               "load_duration": 0, "total_duration": 1_000_000 * max(1, len(text.split()))}
        return out
