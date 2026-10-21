"""Append-only JSONL logging with resume support, plus small helpers."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path


def stable_int(s: str, mod: int = 2**31 - 1) -> int:
    """Process-independent hash (Python's hash() is salted per process)."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % mod


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, cwd=Path(__file__).parent)
        return out.stdout.strip() or "none"
    except Exception:
        return "none"


def append_jsonl(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written last line after a crash
    return rows


def done_keys(path) -> set[str]:
    """Keys of successfully completed generations (error rows are retried on resume)."""
    return {r["key"] for r in read_jsonl(path) if "key" in r and not r.get("error")}


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
