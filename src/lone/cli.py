"""`lone` command line: convert, run, serve, inspect."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _units(args) -> str:
    return "cpu_only" if getattr(args, "cpu_only", False) else getattr(args, "compute_units", "cpu_and_ne")


def cmd_convert(args) -> None:
    from .convert import convert_model

    convert_model(
        model_dir=args.model,
        out_dir=args.out,
        ctx=args.ctx,
        prefill=args.prefill,
        layers_per_model=args.layers_per_model,
        precision=args.precision,
        head_chunks=args.head_chunks,
        decode_only=args.decode_only,
        prefill_only=args.prefill_only,
        check_plan=not args.no_plan,
        compute_units=_units(args),
    )


def cmd_run(args) -> None:
    from .runtime import LoneModel

    model = LoneModel(args.bundle, _units(args))
    if args.chat:
        messages = ([{"role": "system", "content": args.system}] if args.system else []) + [
            {"role": "user", "content": args.prompt}
        ]
        prompt_ids = model.tokenizer.chat(messages)
    else:
        prompt_ids = model.tokenizer.encode(args.prompt)

    first = {"at": None}
    emitted = []
    started = time.perf_counter()

    def on_token(token: int) -> None:
        if first["at"] is None:
            first["at"] = time.perf_counter()
        emitted.append(token)
        if args.stream:
            sys.stdout.write(model.tokenizer.decode([token]))
            sys.stdout.flush()

    ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        on_token=on_token,
    )
    wall = time.perf_counter() - started
    text = model.tokenizer.decode(ids)
    if args.stream:
        sys.stdout.write("\n")
    else:
        print(text)
    prefill_s = (first["at"] or started) - started
    decode_s = max(wall - prefill_s, 1e-9)
    print(f"\n[{len(ids)} tokens | {len(ids) / decode_s:.1f} tok/s | first token {prefill_s * 1e3:.0f} ms | "
          f"{len(prompt_ids)} prompt tokens]", file=sys.stderr)


def cmd_serve(args) -> None:
    import uvicorn

    from .server import create_app

    app = create_app(args.bundle, _units(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_inspect(args) -> None:
    with open(os.path.join(args.bundle, "manifest.json")) as f:
        manifest = json.load(f)
    spec = manifest["spec"]
    print(f"{spec['model_type']}: {spec['num_layers']} layers, hidden {spec['hidden_size']}, "
          f"{spec['num_heads']}/{spec['num_kv_heads']} heads, head_dim {spec['head_dim']}, "
          f"conv {spec['conv_kernel']}, vocab {spec['vocab_size']}")
    print(f"shapes: ctx {manifest['shapes']['ctx']}, prefill {manifest['shapes']['prefill']}, "
          f"precision {manifest['precision']}")
    for chunk in manifest["chunks"]:
        tail = ", ".join(
            f"{tag}: {meta.get('plan', {}).get('devices', 'n/a')}" for tag, meta in chunk["models"].items()
        )
        print(f"  {chunk['name']} layers {chunk['layers']} -> {tail}")
    print(f"  head: {len(manifest['head']['chunks'])} chunks")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lone", description="LFM models on the Apple Neural Engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="convert an HF checkpoint into a LoNE bundle")
    c.add_argument("--model", required=True, help="HF checkpoint directory")
    c.add_argument("--out", required=True, help="bundle output directory")
    c.add_argument("--ctx", type=int, default=2048)
    c.add_argument("--prefill", type=int, default=64)
    c.add_argument("--layers-per-model", type=int, default=1,
                   help="decoder layers per Core ML graph; 1 is the most accurate on ANE")
    c.add_argument("--precision", choices=["fp16", "lut4", "lut6"], default="fp16")
    c.add_argument("--head-chunks", type=int, default=8)
    c.add_argument("--decode-only", action="store_true", help="skip the prefill models (slower prompts)")
    c.add_argument("--prefill-only", action="store_true", help="skip the decode models (prefill timing only)")
    c.add_argument("--no-plan", action="store_true", help="skip the MLComputePlan report")
    c.add_argument("--compute-units", choices=["cpu_and_ne", "cpu_and_gpu", "cpu_only", "all"], default="cpu_and_ne")
    c.set_defaults(func=cmd_convert)

    r = sub.add_parser("run", help="generate from a bundle")
    r.add_argument("--bundle", required=True)
    r.add_argument("--prompt", required=True)
    r.add_argument("--chat", action="store_true", help="render the prompt through the chat template")
    r.add_argument("--system", default=None)
    r.add_argument("--max-new-tokens", type=int, default=256)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--top-k", type=int, default=0)
    r.add_argument("--seed", type=int, default=None)
    r.add_argument("--stream", action="store_true")
    r.add_argument("--cpu-only", action="store_true")
    r.add_argument("--compute-units", choices=["cpu_and_ne", "cpu_and_gpu", "cpu_only", "all"], default="cpu_and_ne")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    s.add_argument("--bundle", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8082)
    s.add_argument("--cpu-only", action="store_true")
    s.add_argument("--compute-units", choices=["cpu_and_ne", "cpu_and_gpu", "cpu_only", "all"], default="cpu_and_ne")
    s.set_defaults(func=cmd_serve)

    i = sub.add_parser("inspect", help="print a bundle's spec and placement")
    i.add_argument("--bundle", required=True)
    i.set_defaults(func=cmd_inspect)

    from .verify import add_verify_parser

    add_verify_parser(sub)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
