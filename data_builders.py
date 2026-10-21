"""Item banks: TruthfulQA loader, determinism prompts, fake-entity probes, JSON extraction tasks.

Everything here is deterministic (seeded) so analysis can regenerate the gold answers
from item ids instead of storing them in every result row.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from io_utils import read_jsonl

# ------------------------------------------------------------------------------------
# TruthfulQA (written by download_datasets.py as JSONL; lists stay real lists, which
# avoids the CSV-stringified-array bug of the original script)
# ------------------------------------------------------------------------------------

def load_truthfulqa(path, n=None) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"{path} is missing or empty. Run: python download_datasets.py")
    return rows[:n] if n else rows


def make_tqa_fixture(path, n=40, seed=0) -> None:
    """Synthetic TruthfulQA-shaped rows for offline tests/dry runs (not real data)."""
    rng = random.Random(seed)
    cats = ["Misconceptions", "Law", "Health", "Science"]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            right = f"The capital of Land{i} is City{i}."
            wrong = f"The capital of Land{i} is Town{i}."
            choices = [right, wrong, f"The capital of Land{i} is Village{i}.", f"Land{i} has no capital."]
            f.write(json.dumps({
                "id": f"tqa-{i:04d}", "question": f"What is the capital of Land{i}?",
                "category": rng.choice(cats), "best_answer": right,
                "correct_answers": [right, f"City{i}"], "incorrect_answers": [wrong, f"Town{i}"],
                "mc1_choices": choices, "mc1_labels": [1, 0, 0, 0],
            }) + "\n")


# ------------------------------------------------------------------------------------
# Determinism prompts: 30 = 6 kinds x 5. Closed prompts should be stable, open ones diverge.
# ------------------------------------------------------------------------------------

def determinism_prompts() -> list[dict]:
    spec = {
        "factual": ("short", [
            "What is the capital of Australia? Answer in one short sentence.",
            "What is the chemical symbol for sodium? Answer in one short sentence.",
            "Who wrote the novel Pride and Prejudice? Answer in one short sentence.",
            "In which year did World War II end? Answer in one short sentence.",
            "What is the largest planet in the Solar System? Answer in one short sentence.",
        ]),
        "arithmetic": ("numeric", [
            "Compute 17 * 23. Briefly show your work and end with the final number.",
            "A shirt costs 40 dollars and is discounted by 15 percent. What is the final price? End with the final number.",
            "What is 1234 + 8765? End your answer with the final number.",
            "A train travels 90 km in 1.5 hours. What is its speed in km/h? End with the final number.",
            "What is the sum of the integers from 1 to 20? End with the final number.",
        ]),
        "code": (None, [
            "Write a Python function that returns the n-th Fibonacci number. Output only code.",
            "Write a Python function that checks whether a string is a palindrome. Output only code.",
            "Write a Python function that merges two sorted lists into one sorted list. Output only code.",
            "Write a Python function that counts word frequencies in a string. Output only code.",
            "Write a Python function that flattens a nested list. Output only code.",
        ]),
        "creative": (None, [
            "Write a two-sentence story about a lighthouse keeper.",
            "Write a short poem about autumn rain (four lines).",
            "Describe a futuristic city in three sentences.",
            "Write a two-sentence opening line for a mystery novel.",
            "Invent a name and a one-sentence description for a new coffee shop.",
        ]),
        "list": (None, [
            "Name five rivers in Africa.",
            "List four programming languages and one use for each.",
            "Give five tips for better sleep.",
            "Name five famous physicists.",
            "List four renewable energy sources.",
        ]),
        "ambiguous": (None, [
            "What is the best way to get there?",
            "Is it better to go with the first option or the second?",
            "How long does it usually take?",
            "Can you explain why that happened?",
            "Which one should I choose?",
        ]),
    }
    out = []
    for kind, (ans_kind, prompts) in spec.items():
        for i, p in enumerate(prompts):
            out.append({"id": f"det-{kind}-{i}", "kind": kind, "answer_kind": ans_kind, "prompt": p})
    return out


# ------------------------------------------------------------------------------------
# Fake-entity probes. Prompts are phrased as REAL questions (no "fictional"/"imaginary").
# The names below are invented, but you must verify each one is truly nonexistent
# (web search / PyPI / arXiv / PubChem) before reporting results: see verification_checklist().
# ------------------------------------------------------------------------------------

_FAKE_PEOPLE = [
    ("Dr. Neral Voss", "molecular astronomy"), ("Professor Ilvane Markham", "computational phonology"),
    ("Dr. Tessaly Brandt", "soil microbiology"), ("Dr. Okonkwo Vell", "quantum chemistry"),
    ("Professor Hedda Ruskin", "medieval Baltic history"), ("Dr. Marlow Quist", "cognitive neuroscience"),
    ("Dr. Anselm Corvane", "marine geophysics"), ("Professor Iria Delacombe", "behavioural economics"),
    ("Dr. Raimund Zeller-Tofte", "plasma physics"), ("Dr. Sunniva Aldercott", "evolutionary linguistics"),
    ("Professor Tomas Verhaegen-Lind", "structural engineering"), ("Dr. Yusra Penhallow", "immunology"),
    ("Dr. Casimir Oduya", "organic photovoltaics"), ("Professor Linnea Marchetti-Roe", "number theory"),
    ("Dr. Bastien Nkemelu", "computational fluid dynamics"),
]
_FAKE_APIS = [
    "pandas.DataFrame.zigzag_merge", "numpy.linalg.triangulate_spectrum", "os.path.mirror_tree",
    "requests.Session.autoreplay", "collections.OrderedBag", "itertools.spiral_chain",
    "json.loads_lenient_strict", "sklearn.cluster.HarmonicKMeans", "torch.nn.functional.lattice_softmax",
    "asyncio.gather_ordered_timeout", "pathlib.Path.deep_symlink_walk", "scipy.stats.kuhlmann_test",
    "matplotlib.pyplot.cascade_subplots", "re.fullsearch_overlapping", "datetime.datetime.business_delta",
]
_FAKE_PAPERS = [
    ('Recursive Attention Lattices for Low-Resource Phonology', 'Verhoeven and Adeyemi', 2019),
    ('Thermal Memory in Cultured Cortical Networks', 'Halvorsen et al.', 2016),
    ('Sparse Gradient Folding for Federated Optimisation', 'Castellanos and Ruiz', 2021),
    ('A Unified Theory of Tidal Resonance in Exoplanet Crusts', 'Imamura and Whitlock', 2018),
    ('Contrastive Lattice Pretraining for Protein Binding', 'Okafor, Lindqvist and Sato', 2022),
    ('Latent Seasonality in Municipal Water Demand', 'Brannigan and Thapa', 2015),
    ('Quasi-Periodic Reward Shaping in Sparse Environments', 'Delacroix et al.', 2020),
    ('Polymer Entanglement Under Oscillating Shear', 'Nakagawa and Fry', 2014),
    ('Neural Symbolic Caching for Theorem Provers', 'Ostrowski and Mbeki', 2023),
    ('Optimal Stopping in Multi-Armed Auctions', 'Haverford and Lim', 2017),
]
_FAKE_COMPOUNDS = [
    "hexafluoro-brenzamide", "tribromo-nexylamine", "cyclo-pentanoxyl diphosphate", "bis(dimethylvorine) chloride",
    "ortho-thalmic acid",
]
_REAL_CONTROLS = [
    "What did Marie Curie contribute to physics and chemistry?",
    "Explain what pandas.DataFrame.merge does and show an example call.",
    'Summarize the main findings of the paper "Attention Is All You Need".',
    "What did Alan Turing contribute to computer science?",
    "Explain what os.path.join does and show an example call.",
    "What is the main use of the compound sodium chloride?",
    "Explain what numpy.linalg.inv does and show an example call.",
    "What did Rosalind Franklin contribute to molecular biology?",
    'Summarize the main findings of the paper "Deep Residual Learning for Image Recognition".',
    "Explain what collections.Counter does and show an example call.",
]


def fake_entity_items() -> list[dict]:
    items = []
    for i, (name, field) in enumerate(_FAKE_PEOPLE):
        items.append({"id": f"fake-person-{i}", "category": "fake_person",
                      "prompt": f"What did {name} contribute to {field}?"})
    for i, api in enumerate(_FAKE_APIS):
        items.append({"id": f"fake-api-{i}", "category": "fake_api",
                      "prompt": f"Explain what {api}() does and show an example call."})
    for i, (title, authors, year) in enumerate(_FAKE_PAPERS):
        items.append({"id": f"fake-paper-{i}", "category": "fake_paper",
                      "prompt": f'Summarize the main findings of the {year} paper "{title}" by {authors}.'})
    for i, c in enumerate(_FAKE_COMPOUNDS):
        items.append({"id": f"fake-compound-{i}", "category": "fake_compound",
                      "prompt": f"What are the main industrial uses of the compound {c}?"})
    for i, q in enumerate(_REAL_CONTROLS):
        items.append({"id": f"real-{i}", "category": "real_control", "prompt": q})
    return items


def verification_checklist() -> str:
    lines = ["Verify each fake item truly does not exist before reporting results:"]
    for it in fake_entity_items():
        if it["category"] != "real_control":
            lines.append(f"  [ ] {it['id']}: {it['prompt']}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------
# JSON extraction tasks (procedural, seeded; gold is regenerated from the id)
# ------------------------------------------------------------------------------------

_VENDORS = ["Northwind Traders", "Bluebird Supplies", "Kestrel Logistics", "Maple & Finch", "Orbit Cloud Ltd",
            "Sandstone Foods", "Quill Print Co"]
_CURRENCIES = ["USD", "EUR", "GBP", "PKR"]
_PRODUCTS = ["Aurora kettle", "Pilot X2 headphones", "TrailLite backpack", "Nimbus desk lamp", "Vertex keyboard",
             "Cedar water bottle"]
_SERVICES = ["auth-service", "payment-gateway", "search-api", "image-resizer", "notifier"]
_FIRST = ["Maria", "Ahmed", "Chen", "Olivia", "Karim", "Sofia", "Daniel", "Aisha", "Lucas", "Fatima"]
_LAST = ["Lopez", "Khan", "Wei", "Bennett", "Haddad", "Rossi", "Okafor", "Malik", "Fischer", "Shah"]
_CITIES = ["Valencia", "Lahore", "Toronto", "Nairobi", "Osaka", "Lyon", "Austin", "Dubai"]
_JOBS = ["nurse", "civil engineer", "data analyst", "chef", "teacher", "electrician", "pharmacist"]
_FUNCS = [("book_flight", "Book {n} seats from {a} to {b}."), ("book_train", "Reserve {n} train tickets from {a} to {b}."),
          ("book_bus", "Get me {n} bus tickets from {a} to {b}.")]


def _schema(props: dict, enums: dict | None = None) -> dict:
    enums = enums or {}
    p = {}
    for k, t in props.items():
        p[k] = {"type": t}
        if k in enums:
            p[k]["enum"] = enums[k]
    return {"type": "object", "properties": p, "required": list(props), "additionalProperties": False}


def json_tasks(n=60, seed=0) -> list[dict]:
    rng = random.Random(seed)
    kinds = ["invoice", "review", "log", "person", "call"]
    tasks = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        if kind == "invoice":
            no = f"INV-{rng.randint(1000, 9999)}"
            vendor, cur = rng.choice(_VENDORS), rng.choice(_CURRENCIES)
            total = round(rng.uniform(20, 5000), 2)
            date = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            ref = f"PO-{rng.randint(1000, 9999)}"
            text = (f"Invoice {no} issued by {vendor} (ref {ref}). Amount due: {cur} {total:.2f}. "
                    f"Please pay by {date}.")
            gold = {"invoice_number": no, "vendor": vendor, "total": total, "currency": cur, "due_date": date}
            schema = _schema({"invoice_number": "string", "vendor": "string", "total": "number",
                              "currency": "string", "due_date": "string"})
        elif kind == "review":
            prod = rng.choice(_PRODUCTS)
            rating = rng.randint(1, 5)
            sent = "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
            phrase = {"positive": "Works great and I love it", "negative": "Disappointing and it broke quickly",
                      "neutral": "It does the job, nothing special"}[sent]
            text = f"I bought the {prod} two weeks ago. {phrase}. I would give it {rating} out of 5 stars."
            gold = {"product": prod, "sentiment": sent, "rating": rating}
            schema = _schema({"product": "string", "sentiment": "string", "rating": "integer"},
                             {"sentiment": ["positive", "negative", "neutral"]})
        elif kind == "log":
            ts = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
            level = rng.choice(["INFO", "WARN", "ERROR"])
            svc = rng.choice(_SERVICES)
            code = rng.choice([200, 201, 400, 404, 500, 503])
            lat = rng.randint(5, 3000)
            text = f"{ts} [{level}] {svc} responded with status {code} in {lat}ms"
            gold = {"timestamp": ts, "level": level, "service": svc, "status_code": code, "latency_ms": lat}
            schema = _schema({"timestamp": "string", "level": "string", "service": "string",
                              "status_code": "integer", "latency_ms": "integer"},
                             {"level": ["INFO", "WARN", "ERROR"]})
        elif kind == "person":
            name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
            age, city, job = rng.randint(21, 70), rng.choice(_CITIES), rng.choice(_JOBS)
            text = f"{name}, {age}, works as a {job} in {city}. She joined the team in 2019."
            gold = {"name": name, "age": age, "city": city, "occupation": job}
            schema = _schema({"name": "string", "age": "integer", "city": "string", "occupation": "string"})
        else:
            fn, tmpl = rng.choice(_FUNCS)
            a, b = rng.sample(_CITIES, 2)
            npax = rng.randint(1, 6)
            text = tmpl.format(n=npax, a=a, b=b)
            gold = {"function": fn, "origin": a, "destination": b, "passengers": npax}
            schema = _schema({"function": "string", "origin": "string", "destination": "string",
                              "passengers": "integer"}, {"function": [f for f, _ in _FUNCS]})
        prompt = ("Extract the requested fields from the text.\n"
                  "Return ONLY a JSON object that matches this JSON schema, with no explanation and no markdown.\n\n"
                  f"Schema:\n{json.dumps(schema)}\n\nText:\n{text}\n\nJSON:")
        tasks.append({"id": f"json-{kind}-{i:03d}", "kind": kind, "prompt": prompt, "schema": schema, "gold": gold})
    return tasks


def load_banks(data_dir: Path, json_n=60, json_seed=0) -> dict:
    """Everything the scorers need, keyed by item id."""
    banks = {"tqa": {}, "json": {t["id"]: t for t in json_tasks(json_n, json_seed)},
             "det": {p["id"]: p for p in determinism_prompts()},
             "fake": {p["id"]: p for p in fake_entity_items()}}
    p = Path(data_dir) / "truthfulqa_sample.jsonl"
    if p.exists():
        banks["tqa"] = {r["id"]: r for r in read_jsonl(p)}
    return banks
