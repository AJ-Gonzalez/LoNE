"""Interactive chat with per-turn stats: rates, context use, memory and engine power.

The model stays in one process, and each turn feeds only the tokens the chat template added since
the last turn, so history is not re-prefilled. Core ML still has to re-specialise on the first call,
which is why the process — not the command — is the unit of warmth.
"""

from __future__ import annotations

import sys
import time

import typer

from .cli_options import ComputeUnits
from .metrics import ANECounters, PowerCounters, memory
from .runtime import LoneError, LoneModel

EXIT = {"/quit", "/exit", "quit", "exit", "q"}
HELP = "commands: /reset clears the conversation, /quit exits"


def common_prefix(a: list[int], b: list[int]) -> int:
    """Length of the shared prefix of two token lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _rate(tokens: int, seconds: float) -> str:
    if seconds <= 0:
        return "—"
    return f"{tokens / seconds:.1f} tok/s"


def _gb(value: float) -> str:
    return f"{value / 2**30:.2f} GB"


def format_stats(*, prefill_tokens: int, prefill_s: float, prefill_w: float | None,
                 decode_tokens: int, decode_s: float, decode_w: float | None,
                 used: int, ctx: int, mem: dict, ane_gb: float | None) -> list[str]:
    """Two lines: rates and power, then context, memory and engine traffic."""
    pre = f"prefill {prefill_tokens} tok {prefill_s:.2f} s {_rate(prefill_tokens, prefill_s)}"
    if prefill_w is not None:
        pre += f" {prefill_w:.2f} W"
    dec = f"decode {decode_tokens} tok {decode_s:.2f} s {_rate(decode_tokens, decode_s)}"
    if decode_w is not None:
        dec += f" {decode_w:.2f} W"

    share = f" ({used / ctx * 100:.1f}%)" if ctx else ""
    parts = [f"ctx {used}/{ctx}{share}"]
    if mem:
        parts.append(f"mem {_gb(mem['resident'])} rss")
        if "free" in mem:
            parts.append(f"{_gb(mem['free'])} free")
    if ane_gb is not None:
        parts.append(f"ane io {ane_gb:.2f} GB")
    return [f"  {pre} · {dec}", "  " + " · ".join(parts)]


def register_chat(app: typer.Typer) -> None:
    @app.command(help="chat in the terminal; one line of stats per turn")
    def chat(
        bundle: str = typer.Option(..., "--bundle"),
        system: str | None = typer.Option(None, "--system"),
        temperature: float = typer.Option(0.7, "--temperature"),
        top_p: float = typer.Option(0.95, "--top-p"),
        top_k: int = typer.Option(0, "--top-k"),
        max_new_tokens: int = typer.Option(256, "--max-new-tokens"),
        wire_caches: bool = typer.Option(False, "--wire-caches", help="wire the caches with mlock (macOS)"),
        cpu_only: bool = typer.Option(False, "--cpu-only"),
        compute_units: ComputeUnits = typer.Option(ComputeUnits.cpu_and_ne, "--compute-units"),
        stats: bool = typer.Option(True, "--stats/--no-stats"),
    ) -> None:
        from .cli import _units

        model = LoneModel(bundle, _units(cpu_only, compute_units), wire_caches=wire_caches)
        tok = model.tokenizer
        try:
            import readline  # noqa: F401  (line editing and history when stdin is a tty)
        except ImportError:  # pragma: no cover - readline is stdlib on macOS
            pass

        power = PowerCounters()
        ane = ANECounters()
        memory()  # prime the mach calls before the first timed turn
        system_message = [{"role": "system", "content": system}] if system else []
        messages: list[dict] = list(system_message)
        fed: list[int] = []
        print(f"{bundle}  ctx {model.ctx}  /help for commands")

        interactive = sys.stdin.isatty()

        while True:
            try:
                line = input("you> " if interactive else "")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print()
                continue
            if not line.strip():
                continue
            command = line.strip().lower()
            if command in EXIT:
                break
            if command == "/help":
                print(HELP)
                continue
            if command == "/reset":
                model.reset()
                fed, messages = [], list(system_message)
                print("(reset)")
                continue

            messages.append({"role": "user", "content": line})
            try:
                rendered = tok.chat(messages)
            except LoneError as exc:
                print(f"error: {exc}")
                messages.pop()
                continue
            keep = common_prefix(fed, rendered)
            if keep < len(fed):
                # The template re-rendered history into different tokens; start the caches over.
                model.reset()
                keep = 0
            fresh = rendered[keep:]
            if model.pos + len(fresh) >= model.ctx:
                print(f"(context full: {model.pos} + {len(fresh)} >= {model.ctx}; /reset to clear)")
                messages.pop()
                continue

            before_power = power.read() if power.ok else {}
            before_ane = ane.read() if ane.ok else {}
            marks: dict = {"first": None, "power": {}}

            def on_token(token: int) -> None:
                if marks["first"] is None:
                    marks["first"] = time.perf_counter()
                    if power.ok:
                        marks["power"] = power.read()
                sys.stdout.write(tok.decode([token]))
                sys.stdout.flush()

            if interactive:
                print("lone> ", end="")
                sys.stdout.flush()
            started = time.perf_counter()
            if fresh:
                model.feed(fresh)
            out = model.generate_next(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                on_token=on_token,
            )
            first = marks["first"] or time.perf_counter()
            # The final sampled token is not fed when generation stops at the limit; feed it so the
            # caches match `rendered + out` and the next turn can extend the prefix.
            missing = (len(rendered) + len(out)) - model.pos
            if missing > 0:
                model.feed(out[-missing:])
            ended = time.perf_counter()
            fed = rendered + out
            messages.append({"role": "assistant", "content": tok.decode(out)})
            print()

            if stats:
                after_power = power.read() if power.ok else {}
                after_ane = ane.read() if ane.ok else {}
                prefill_w = decode_w = None
                if power.ok and marks["power"] and before_power:
                    prefill_w = power.watts(before_power, marks["power"], first - started).get("ANE")
                    decode_w = power.watts(marks["power"], after_power, ended - first).get("ANE")
                ane_gb = None
                if ane.ok and before_ane and after_ane:
                    delta = ane.delta(before_ane, after_ane)
                    ane_gb = sum(delta.values()) / 1e9
                for row in format_stats(
                    prefill_tokens=len(fresh),
                    prefill_s=first - started,
                    prefill_w=prefill_w,
                    decode_tokens=max(len(out) - 1, 0),
                    decode_s=max(ended - first, 0.0),
                    decode_w=decode_w,
                    used=model.pos,
                    ctx=model.ctx,
                    mem=memory(),
                    ane_gb=ane_gb,
                ):
                    print(row)
