"""`lone` command line: convert, run, serve, inspect, verify.

Typer wraps the same core functions; option names and defaults are unchanged from the argparse
version. `main()` is kept as the console-script target.
"""

from __future__ import annotations

import json
import os
import sys
import time

import typer

from .cli_options import ComputeUnits, Precision

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="LFM models on the Apple Neural Engine",
)


def _units(cpu_only: bool, compute_units: ComputeUnits) -> str:
    return "cpu_only" if cpu_only else compute_units.value


def _report_wired(wired_bytes: int) -> None:
    if wired_bytes:
        print(f"wired {wired_bytes / 2**20:.0f} MiB of caches", file=sys.stderr)
    else:
        print("cache wiring unavailable (needs macOS mlock); continuing unwired", file=sys.stderr)


@app.command(help="convert an HF checkpoint into a LoNE bundle")
def convert(
    model: str = typer.Option(..., "--model", help="HF checkpoint directory"),
    out: str = typer.Option(..., "--out", help="bundle output directory"),
    ctx: int = typer.Option(2048, "--ctx"),
    prefill: int = typer.Option(64, "--prefill"),
    layers_per_model: int = typer.Option(
        4, "--layers-per-model", help="decoder layers per Core ML graph; fewer graphs are faster, 4 verifies at parity"
    ),
    precision: Precision = typer.Option(Precision.fp16, "--precision"),
    head_chunks: int = typer.Option(
        0, "--head-chunks", help="vocabulary slices for the LM head; 0 sizes each graph to ~128 MB"
    ),
    ring: bool = typer.Option(
        True, "--ring/--no-ring",
        help="ring-buffer sliding layers to their window instead of the full ctx (default on)",
    ),
    decode_only: bool = typer.Option(False, "--decode-only", help="skip the prefill models (slower prompts)"),
    prefill_only: bool = typer.Option(False, "--prefill-only", help="skip the decode models (prefill timing only)"),
    spec_drafts: int = typer.Option(
        0, "--spec-drafts",
        help="convert a verify shape for n-gram speculative decoding: prefill becomes drafts+1 and "
             "the head is traced at that length (0 = off)",
    ),
    plan: bool = typer.Option(True, "--plan/--no-plan", help="report the MLComputePlan placement per model"),
    compute_units: ComputeUnits = typer.Option(ComputeUnits.cpu_and_ne, "--compute-units"),
) -> None:
    from .convert import convert_model

    convert_model(
        model_dir=model,
        out_dir=out,
        ctx=ctx,
        prefill=prefill,
        layers_per_model=layers_per_model,
        precision=precision.value,
        head_chunks=head_chunks,
        ring=ring,
        decode_only=decode_only,
        prefill_only=prefill_only,
        spec_drafts=spec_drafts,
        check_plan=plan,
        compute_units=compute_units.value,
    )


@app.command(help="generate from a bundle")
def run(
    bundle: str = typer.Option(..., "--bundle"),
    prompt: str = typer.Option(..., "--prompt"),
    chat: bool = typer.Option(False, "--chat", help="render the prompt through the chat template"),
    system: str | None = typer.Option(None, "--system"),
    max_new_tokens: int = typer.Option(256, "--max-new-tokens"),
    temperature: float = typer.Option(0.0, "--temperature"),
    top_p: float = typer.Option(1.0, "--top-p"),
    top_k: int = typer.Option(0, "--top-k"),
    seed: int | None = typer.Option(None, "--seed"),
    stream: bool = typer.Option(False, "--stream"),
    cpu_only: bool = typer.Option(False, "--cpu-only"),
    compute_units: ComputeUnits = typer.Option(ComputeUnits.cpu_and_ne, "--compute-units"),
    wire_caches: bool = typer.Option(
        False, "--wire-caches", help="wire the KV and conv caches with mlock so they cannot be paged out (macOS)"
    ),
    spec_ngram: bool = typer.Option(
        False, "--spec-ngram/--no-spec-ngram",
        help="n-gram speculative decoding (llama.cpp ngram-simple); needs a bundle converted with --spec-drafts",
    ),
    spec_n: int = typer.Option(12, "--spec-n", help="lookup n-gram length for --spec-ngram"),
    spec_m: int = typer.Option(16, "--spec-m", help="maximum draft length for --spec-ngram"),
    spec_drafts: int = typer.Option(
        0, "--spec-drafts", help="cap on verified drafts per step for --spec-ngram (0 = the bundle's)"
    ),
) -> None:
    from .runtime import LoneModel
    from .spec import NgramProposer

    model = LoneModel(bundle, _units(cpu_only, compute_units), wire_caches=wire_caches)
    if wire_caches:
        _report_wired(model.wired_bytes)
    if chat:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        prompt_ids = model.tokenizer.chat(messages)
    else:
        prompt_ids = model.tokenizer.encode(prompt)

    first = {"at": None}
    emitted = []
    started = time.perf_counter()

    def on_token(token: int) -> None:
        if first["at"] is None:
            first["at"] = time.perf_counter()
        emitted.append(token)
        if stream:
            sys.stdout.write(model.tokenizer.decode([token]))
            sys.stdout.flush()

    proposer = NgramProposer(spec_n, spec_m) if spec_ngram else None
    ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        on_token=on_token,
        proposer=proposer,
        max_drafts=spec_drafts,
    )
    wall = time.perf_counter() - started
    text = model.tokenizer.decode(ids)
    if stream:
        sys.stdout.write("\n")
    else:
        print(text)
    prefill_s = (first["at"] or started) - started
    decode_s = max(wall - prefill_s, 1e-9)
    decode_tokens = max(len(ids) - 1, 0)
    rate = decode_tokens / decode_s if decode_tokens else 0.0
    spec = ""
    stats = model.spec_stats
    if proposer is not None:
        drafted = stats.get("drafted", 0)
        hit = f" ({stats.get('accepted', 0) / drafted:.0%})" if drafted else ""
        spec = (f" | spec {stats.get('accepted', 0)}/{drafted} drafts{hit}"
                f" in {stats.get('passes', 0)} passes")
    print(f"\n[{len(ids)} tokens | {rate:.1f} tok/s | first token {prefill_s * 1e3:.0f} ms | "
          f"{len(prompt_ids)} prompt tokens{spec}]", file=sys.stderr)


@app.command(help="OpenAI-compatible HTTP server")
def serve(
    bundle: str = typer.Option(..., "--bundle"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8082, "--port"),
    cpu_only: bool = typer.Option(False, "--cpu-only"),
    compute_units: ComputeUnits = typer.Option(ComputeUnits.cpu_and_ne, "--compute-units"),
    wire_caches: bool = typer.Option(
        False, "--wire-caches", help="wire the KV and conv caches with mlock so they cannot be paged out (macOS)"
    ),
) -> None:
    import uvicorn

    from .server import create_app

    app = create_app(bundle, _units(cpu_only, compute_units), wire_caches=wire_caches)
    if wire_caches:
        _report_wired(app.state.wired_bytes)
    uvicorn.run(app, host=host, port=port, log_level="info")


@app.command("inspect", help="print a bundle's spec and placement")
def inspect_cmd(bundle: str = typer.Option(..., "--bundle")) -> None:
    with open(os.path.join(bundle, "manifest.json")) as f:
        manifest = json.load(f)
    spec = manifest["spec"]
    print(f"{spec['model_type']}: {spec['num_layers']} layers, hidden {spec['hidden_size']}, "
          f"{spec['num_heads']}/{spec['num_kv_heads']} heads, head_dim {spec['head_dim']}, "
          f"conv {spec['conv_kernel']}, vocab {spec['vocab_size']}")
    print(f"shapes: ctx {manifest['shapes']['ctx']}, prefill {manifest['shapes']['prefill']}, "
          f"precision {manifest['precision']}")
    drafts = manifest.get("speculative", {}).get("drafts", 0)
    if drafts:
        print(f"spec: n-gram verify shape, {drafts} drafts/pass, head {manifest['head'].get('seq')} rows")
    window = spec.get("sliding_window")
    if manifest.get("ring") and window:
        print(f"ring: sliding cache {min(window, manifest['shapes']['ctx'])} rows (window {window})")
    for chunk in manifest["chunks"]:
        tail = ", ".join(
            f"{tag}: {meta.get('plan', {}).get('devices', 'n/a')}" for tag, meta in chunk["models"].items()
        )
        print(f"  {chunk['name']} layers {chunk['layers']} -> {tail}")
    print(f"  head: {len(manifest['head']['chunks'])} chunks")


from .repl import register_chat  # noqa: E402  (needs `app` above)
from .verify import register_verify  # noqa: E402  (needs `app` above)

register_chat(app)
register_verify(app)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
