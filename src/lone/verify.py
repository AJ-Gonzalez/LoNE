"""End-to-end parity check: one bundle against the HF reference.

Two phases keep the 8 GB machines honest: the HF checkpoint (3.3 GB for LFM3) and the Core ML
bundle never live in the same process. `lone verify` spawns the reference phase as a subprocess,
caches its hidden states in an `.npz`, then runs the bundle and prints the drift curve.

The bundle is always driven through its decode models, one token at a time, so the comparison is
against the path that runs on the ANE at inference (prefill graphs are a different shape and are
not covered here).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import typer

from .cli_options import ComputeUnits

BAR = 0.999


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------- reference phase
def hf_reference(model_dir: str, ids: list[int], out_path: str, max_new_tokens: int = 0) -> dict:
    """Run the HF checkpoint on `ids` and save hidden states, last-token logits, greedy tokens."""
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16, local_files_only=True, trust_remote_code=True
    )
    model.eval()
    # transformers records the model output (post-final-norm) as the last hidden state. Hook the
    # final norm so the curve stops at the last decoder layer, like the bundle's last chunk.
    norm = next((m for name, m in model.named_modules() if name.endswith("embedding_norm")), None)
    captured: dict[str, "torch.Tensor"] = {}
    handle = None
    if norm is not None:
        def grab(_module, args):
            captured.setdefault("pre", args[0].detach())

        handle = norm.register_forward_pre_hook(grab)
    try:
        with torch.no_grad():
            out = model(torch.tensor([ids]), output_hidden_states=True)
    finally:
        if handle is not None:
            handle.remove()
    states = list(out.hidden_states)
    if "pre" in captured:
        states[-1] = captured["pre"]
    hidden = np.stack([h[0].to(torch.float32).numpy() for h in states])  # (L+1, seq, H)
    logits = out.logits[0, -1].to(torch.float32).numpy()
    argmax = out.logits[0, -1].argmax().item()
    generated = np.asarray([], dtype=np.int64)
    if max_new_tokens > 0:
        with torch.no_grad():
            sequence = model.generate(torch.tensor([ids]), do_sample=False,
                                      max_new_tokens=max_new_tokens, use_cache=True)
        generated = np.asarray(sequence[0, len(ids):].tolist(), dtype=np.int64)
    np.savez(out_path, hidden=hidden, logits=logits, ids=np.asarray(ids, dtype=np.int64),
             argmax=np.asarray(argmax), generated=generated)
    return {"hidden": hidden, "logits": logits, "argmax": int(argmax), "generated": generated.tolist()}


# ----------------------------------------------------------------------------- bundle phase
def bundle_capture(bundle_dir: str, ids: list[int], compute_units: str = "cpu_and_ne",
                   max_new_tokens: int = 0) -> dict:
    """Run the bundle's decode path over `ids` and capture the hidden state after every chunk."""
    from .runtime import LoneModel

    model = LoneModel(bundle_dir, compute_units)
    model.reset()
    per_chunk: list[list[np.ndarray]] = [[] for _ in model._chunks]
    embedding = None
    logits = None
    for i, token in enumerate(ids):
        captured: list[np.ndarray] = []
        embedding = model._embed_tokens([token])[0, 0].astype(np.float32)
        model._forward([token], "decode", capture=captured)
        for chunk_index, hidden in enumerate(captured):
            per_chunk[chunk_index].append(hidden[-1].astype(np.float32))
        logits = model._logits()
    generated = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0) if max_new_tokens > 0 else []
    return {
        "embedding": embedding,
        "hidden": [np.stack(chunk) for chunk in per_chunk],  # per chunk: (seq, H)
        "logits": logits,
        "argmax": int(np.argmax(logits)),
        "layers": [chunk["layers"] for chunk in model._chunks],
        "generated": generated,
    }


# ------------------------------------------------------------------------------- comparison
def compare(reference: dict, bundle: dict) -> dict:
    """Per-layer hidden cosine for the last position, plus logits cosine / argmax agreement."""
    ref_hidden = reference["hidden"]  # (L+1, seq, H); hidden[0] is the embedding output
    rows = []
    embedding = cosine(ref_hidden[0, -1], bundle["embedding"])
    for chunk_pos, chunk_hidden in enumerate(bundle["hidden"]):
        layer = bundle["layers"][chunk_pos][-1]
        ref = ref_hidden[layer + 1, -1]
        rows.append({"layer": layer, "cosine": cosine(ref, chunk_hidden[-1]),
                     "max_abs": float(np.max(np.abs(ref.astype(np.float32) - chunk_hidden[-1])))})
    logits = cosine(reference["logits"], bundle["logits"])
    generated_ref = reference.get("generated", [])
    generated_bundle = bundle.get("generated", [])
    result = {
        "embedding": embedding,
        "layers": rows,
        "hidden_min": min(row["cosine"] for row in rows),
        "logits": logits,
        "argmax_reference": int(reference["argmax"]),
        "argmax_bundle": int(bundle["argmax"]),
        "argmax_match": int(reference["argmax"]) == int(bundle["argmax"]),
    }
    if generated_ref and generated_bundle:
        first_diff = next((i for i, (a, b) in enumerate(zip(generated_ref, generated_bundle)) if a != b), None)
        if first_diff is None and len(generated_ref) != len(generated_bundle):
            first_diff = min(len(generated_ref), len(generated_bundle))
        result.update({
            "generated_reference": [int(x) for x in generated_ref],
            "generated_bundle": [int(x) for x in generated_bundle],
            "generated_divergence": first_diff,
        })
    return result


def _print_report(result: dict, tokens: int) -> None:
    print(f"prompt: {tokens} tokens")
    print(f"logits cosine {result['logits']:.7f}  argmax {result['argmax_bundle']} "
          f"({'match' if result['argmax_match'] else 'MISMATCH, reference ' + str(result['argmax_reference'])})")
    print(f"embedding cosine {result['embedding']:.7f}")
    if "generated_divergence" in result:
        n = len(result["generated_reference"])
        d = result["generated_divergence"]
        print(f"greedy tokens: {n} reference, {len(result['generated_bundle'])} bundle, "
              f"{'identical' if d is None else f'first divergence at index {d}'}")
    print(f"{'layer':>5}  {'hidden cos':>10}  {'max abs':>9}")
    for row in result["layers"]:
        print(f"{row['layer']:>5}  {row['cosine']:>10.6f}  {row['max_abs']:>9.3f}")
    print(f"hidden min {result['hidden_min']:.6f}  |  logits {result['logits']:.7f} "
          f"({'PASS' if result['logits'] >= BAR else 'BELOW'} the {BAR} bar)")


# ------------------------------------------------------------------------------------- cli
def _reference_subprocess(model_dir: str, ids: list[int], out_path: str, max_new_tokens: int = 0) -> None:
    code = "from lone.verify import _reference_main; import sys; _reference_main(sys.argv[1:])"
    cmd = [sys.executable, "-c", code, model_dir, json.dumps(ids), out_path, str(max_new_tokens)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"HF reference phase failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")


def _reference_main(argv: list[str]) -> None:
    model_dir, ids_json, out_path, max_new_tokens = argv
    hf_reference(model_dir, json.loads(ids_json), out_path, int(max_new_tokens))


def _verify(
    bundle: str,
    prompt: str,
    model: str | None,
    reference: str | None,
    save_reference: str | None,
    chat: bool,
    system: str | None,
    min_cos: float,
    max_new_tokens: int,
    allow_below: bool,
    as_json: bool,
    compute_units: str,
) -> None:
    from .runtime import BundleTokenizer

    tok = BundleTokenizer(os.path.join(bundle, "tokenizer"))
    if chat:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        ids = tok.chat(messages)
    else:
        ids = tok.encode(prompt)

    reference_path = reference if reference and os.path.exists(reference) else None
    if reference_path is None:
        if not model:
            raise SystemExit("pass --model to compute the HF reference or --reference to reuse one")
        reference_path = reference or save_reference
        if reference_path is None:
            fd, reference_path = tempfile.mkstemp(suffix=".npz")
            os.close(fd)
        else:
            print(f"writing HF reference to {reference_path}")
        _reference_subprocess(model, ids, reference_path, max_new_tokens)

    reference_data = np.load(reference_path)
    reference = {
        "hidden": reference_data["hidden"],
        "logits": reference_data["logits"],
        "argmax": int(reference_data["argmax"]),
        "generated": reference_data["generated"].tolist() if "generated" in reference_data else [],
    }
    saved_ids = reference_data["ids"].tolist()
    if saved_ids != ids:
        raise SystemExit("reference was computed on different tokens; regenerate it")

    bundle_data = bundle_capture(bundle, ids, compute_units, max_new_tokens)
    result = compare(reference, bundle_data)
    if as_json:
        print(json.dumps(result, indent=2))
    else:
        _print_report(result, len(ids))
    if not allow_below and result["logits"] < min_cos:
        raise SystemExit(1)


def register_verify(app: typer.Typer) -> None:
    @app.command("verify", help="compare a bundle against the HF reference")
    def verify(
        bundle: str = typer.Option(..., "--bundle"),
        prompt: str = typer.Option(..., "--prompt"),
        model: str | None = typer.Option(None, "--model", help="HF checkpoint directory used as the reference"),
        reference: str | None = typer.Option(None, "--reference", help="reuse a cached reference .npz"),
        save_reference: str | None = typer.Option(None, "--save-reference", help="write the reference .npz here"),
        chat: bool = typer.Option(False, "--chat", help="render the prompt through the chat template"),
        system: str | None = typer.Option(None, "--system"),
        min_cos: float = typer.Option(BAR, "--min-cos", help="logits cosine required to exit 0"),
        max_new_tokens: int = typer.Option(
            0, "--max-new-tokens", help="also compare this many greedy tokens against the HF reference"
        ),
        allow_below: bool = typer.Option(False, "--allow-below", help="report but do not fail below the bar"),
        as_json: bool = typer.Option(False, "--json"),
        cpu_only: bool = typer.Option(False, "--cpu-only"),
        compute_units: ComputeUnits = typer.Option(ComputeUnits.cpu_and_ne, "--compute-units"),
    ) -> None:
        _verify(
            bundle=bundle,
            prompt=prompt,
            model=model,
            reference=reference,
            save_reference=save_reference,
            chat=chat,
            system=system,
            min_cos=min_cos,
            max_new_tokens=max_new_tokens,
            allow_below=allow_below,
            as_json=as_json,
            compute_units="cpu_only" if cpu_only else compute_units.value,
        )
