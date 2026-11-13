"""A/B n-gram speculative decoding on one LoNE bundle.

Both arms run in one process, alternating per repeat, because this machine's decode rate is
residency-sensitive: a cold bundle measured 201 ms/token against 84 ms warm for the same graphs.
The baseline uses the decode shape (one token per pass); the speculative arm uses the verify shape
(`--spec-drafts` at conversion). Prefill is excluded from every rate.

    python bench/spec_ab.py --bundle /path/to/bundle --workload copy --tokens 128
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from lone.metrics import PowerCounters  # noqa: E402
from lone.runtime import LoneModel  # noqa: E402
from lone.spec import NgramProposer  # noqa: E402

FILE = """\
def load_config(path):
    with open(path) as handle:
        return json.load(handle)


def save_config(path, config):
    with open(path, "w") as handle:
        json.dump(config, handle, indent=2)


def merge_config(base, override):
    merged = dict(base)
    merged.update(override)
    return merged


def get_value(config, key, default=None):
    return config.get(key, default)


def set_value(config, key, value):
    config[key] = value
    return config
"""

WORKLOADS = {
    # Repetition: the model is asked to reproduce a block it has already seen, which is what the
    # n-gram proposer drafts from.
    "copy": (
        "Here is a Python module:\n\n" + FILE + "\nReturn the module again, unchanged, then add "
        "a function `del_value(config, key)` that removes the key and returns the config.\n\n"
        "```python\n"
    ),
    # Repetition without an instruction: the prompt already holds two identical copies, so a raw
    # completion continues with a third and the 12-gram lookup fires immediately.
    "repeat": FILE + "\n# ---- identical copy ----\n\n" + FILE + "\n# ---- identical copy ----\n\n",
    # Control: no long verbatim span to copy, so the proposer should rarely fire.
    "novel": (
        "Write a short story about a lighthouse keeper who finds a message in a bottle. "
        "Start with: The lamp had been cold for three nights when\n"
    ),
}


def run_arm(model, ids, tokens, proposer, max_drafts, temperature, top_k, top_p):
    power = PowerCounters()
    before = power.read() if power.ok else {}
    started = time.perf_counter()
    first = {"at": None}

    def on_token(_):
        if first["at"] is None:
            first["at"] = time.perf_counter()

    out = model.generate(
        ids, max_new_tokens=tokens, temperature=temperature, top_p=top_p, top_k=top_k,
        on_token=on_token, proposer=proposer, max_drafts=max_drafts,
    )
    wall = time.perf_counter() - started
    after = power.read() if power.ok else {}
    prefill_s = (first["at"] or started) - started
    decode_s = max(wall - prefill_s, 1e-9)
    decode_w = power.watts(before, after, wall).get("ANE", 0.0) if before and after else 0.0
    return {
        "ane_w": decode_w,
        "tokens": out,
        "decode_tokens": max(len(out) - 1, 0),
        "decode_s": decode_s,
        "rate": max(len(out) - 1, 0) / decode_s,
        "stats": dict(model.spec_stats),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--workload", choices=sorted(WORKLOADS), default="copy")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1,
                    help="unmeasured passes per arm first: loads the prefill graphs and warms the weights")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--spec-n", type=int, default=12)
    ap.add_argument("--spec-m", type=int, default=16)
    ap.add_argument("--spec-drafts", type=int, default=0, help="0 = the bundle's")
    ap.add_argument("--compute-units", default="cpu_and_ne")
    ap.add_argument("--chat", action="store_true", help="render the workload through the chat template")
    args = ap.parse_args()

    model = LoneModel(args.bundle, args.compute_units)
    proposer = NgramProposer(args.spec_n, args.spec_m)
    drafts = args.spec_drafts or model.spec_drafts
    if drafts < 1:
        raise SystemExit("bundle has no verify shape: reconvert with --spec-drafts")

    prompt = WORKLOADS[args.workload]
    ids = model.tokenizer.chat([{"role": "user", "content": prompt}]) if args.chat else \
        model.tokenizer.encode(prompt)
    if len(ids) >= model.ctx:
        raise SystemExit(f"prompt is {len(ids)} tokens, ctx is {model.ctx}")

    for _ in range(args.warmup):
        run_arm(model, ids, args.tokens, None, 0, args.temperature, args.top_k, args.top_p)
        run_arm(model, ids, args.tokens, proposer, drafts, args.temperature, args.top_k, args.top_p)

    base, spec = [], []
    for _ in range(args.repeats):
        base.append(run_arm(model, ids, args.tokens, None, 0, args.temperature, args.top_k, args.top_p))
        spec.append(run_arm(model, ids, args.tokens, proposer, drafts, args.temperature,
                            args.top_k, args.top_p))

    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    agree = np.mean([a == b for a, b in zip(base[-1]["tokens"], spec[-1]["tokens"])])
    drafted = sum(r["stats"].get("drafted", 0) for r in spec)
    accepted = sum(r["stats"].get("accepted", 0) for r in spec)
    passes = sum(r["stats"].get("passes", 0) for r in spec)
    report = {
        "bundle": args.bundle,
        "workload": args.workload,
        "prompt_tokens": len(ids),
        "tokens": args.tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "spec_n": args.spec_n,
        "spec_m": args.spec_m,
        "max_drafts": drafts,
        "base_tok_s": mean(base, "rate"),
        "spec_tok_s": mean(spec, "rate"),
        "speedup": mean(spec, "rate") / mean(base, "rate"),
        "drafted": drafted,
        "accepted": accepted,
        "acceptance": accepted / drafted if drafted else 0.0,
        "verify_passes": passes,
        "tokens_per_verify": (accepted + passes) / passes if passes else 0.0,
        "agreement": float(agree),
        "base_runs": [round(r["rate"], 2) for r in base],
        "spec_runs": [round(r["rate"], 2) for r in spec],
        "base_w": mean(base, "ane_w"),
        "spec_w": mean(spec, "ane_w"),
        "base_mj_per_token": 1000.0 * mean(base, "ane_w") / mean(base, "rate") if mean(base, "rate") else 0.0,
        "spec_mj_per_token": 1000.0 * mean(spec, "ane_w") / mean(spec, "rate") if mean(spec, "rate") else 0.0,
        "sample": model.tokenizer.decode(spec[-1]["tokens"][:48]),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
