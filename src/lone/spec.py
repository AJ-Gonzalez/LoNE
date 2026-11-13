"""N-gram speculative decoding: draft proposal, sampling support and the accept loop.

`NgramProposer` mirrors llama.cpp's `ngram-simple` (`common/ngram-map.cpp`): no index is built.
Every step rescans the token history for the most recent earlier occurrence of the current n-gram
and drafts the m-gram that followed it, so a repeated stretch of text is drafted from itself. It
needs `n + m + 1` tokens of history, ignores a match at position 0, and discards drafts shorter
than `n` — all three quirks are llama.cpp's, kept so the two runtimes can be compared directly.

The accept rule is exact for any draft distribution, not just greedy: a draft token is accepted
with the target's own probability for it, and a rejected position draws from the target minus that
token (speculative sampling with a point-mass proposal). Greedy is the same rule with
`p(draft) ∈ {0, 1}`, so one code path covers both and the accepted sequence is identical to
unspeculated greedy decode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Ranks past this can only be sampled when the distribution is flat enough that top-p keeps most
# of the vocabulary; the prefilter detects that and falls back to the full vector.
_SAMPLE_CANDIDATES = 1024


@dataclass(frozen=True)
class NgramProposer:
    """llama.cpp `ngram-simple`: `size_n` lookup length, `size_m` maximum draft length."""

    size_n: int = 12
    size_m: int = 16

    def propose(self, tokens, sampled: int) -> list[int]:
        """Draft tokens following the most recent earlier match of the last `size_n` tokens.

        `tokens` is the history without `sampled`, the token that was just drawn.
        """
        cur = len(tokens)
        if cur <= self.size_n + self.size_m + 1:
            return []
        pattern = list(tokens[cur - self.size_n + 1 :]) + [sampled]
        match = 0
        for j in range(cur - self.size_n - 1, 0, -1):
            if tokens[j : j + self.size_n] == pattern:
                match = j
                break
        if not match:
            return []
        copy_max = min(self.size_m, cur - (match + self.size_n))
        if copy_max < self.size_n:
            return []
        return list(tokens[match + self.size_n : match + self.size_n + copy_max])


# ------------------------------------------------------------------ sampling support
def distribution(x: np.ndarray, top_p: float = 1.0, top_k: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Token ids and probabilities of the sampling support for temperature-scaled logits `x`.

    Sorting the whole vocabulary per token costs 9 ms at 128k, but only the head of the
    distribution can ever be drawn: `top_k` is a hard cut and `top_p < 1` keeps a few hundred
    ranks in practice. The support is found by `argpartition` and only that slice is sorted. The
    probabilities are the true ones (divided by the full-vocabulary mass), so the kept set matches
    the vocabulary-wide computation; when top-p would reach beyond the slice, the full vector is
    used instead.
    """
    size = x.size

    if top_k and top_k < size:
        # Keep every token tied at the cut, as a vocabulary-wide mask does.
        cut = np.partition(x, -top_k)[-top_k]
        cand = np.flatnonzero(x >= cut)
        order = np.argsort(-x[cand])
        kept, probs = _top_p(_softmax(x[cand][order].astype(np.float64)), top_p)
        return cand[order[kept]], probs

    if top_p < 1.0 and size > _SAMPLE_CANDIDATES:
        e = np.exp(x.astype(np.float64) - float(x.max()))
        total = e.sum()
        cand = np.argpartition(e, -_SAMPLE_CANDIDATES)[-_SAMPLE_CANDIDATES:]
        probs = e[cand] / total
        order = np.argsort(-probs)
        kept, normalised = _top_p(probs[order], top_p)
        if kept.size < order.size:
            return cand[order[kept]], normalised
        # Top-p needs ranks outside the slice: the full distribution is already exponentiated.
        probs = e / total
    else:
        probs = _softmax(x.astype(np.float64))

    if top_p >= 1.0:
        return np.arange(size), probs
    order = np.argsort(-probs)
    kept, normalised = _top_p(probs[order], top_p)
    return order[kept], normalised


def _top_p(sorted_probs: np.ndarray, top_p: float) -> tuple[np.ndarray, np.ndarray]:
    """Indices and renormalised probabilities of the top-p set of `sorted_probs` (descending)."""
    if top_p >= 1.0:
        return np.arange(sorted_probs.size), sorted_probs
    keep = np.cumsum(sorted_probs) <= top_p
    keep[0] = True
    probs = sorted_probs[keep]
    return np.flatnonzero(keep), probs / probs.sum()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def draw(logits: np.ndarray, temperature: float = 0.0, top_p: float = 1.0, top_k: int = 0,
         rng: np.random.Generator | None = None) -> int:
    """One token from `logits`: argmax at temperature 0, otherwise `distribution` then one draw."""
    if temperature <= 0:
        return int(np.argmax(logits))
    if rng is None:
        rng = np.random.default_rng()
    ids, probs = distribution(logits / np.float32(temperature), top_p, top_k)
    return int(rng.choice(ids, p=probs))


def acceptance(temperature: float, top_p: float, top_k: int, rng: np.random.Generator):
    """Build the per-position check: `(accepted, corrected token or None)`.

    `corrected` is only produced on a rejection, and is drawn from the target distribution with the
    draft token removed — the residual of a point-mass proposal, renormalised.
    """

    def check(row: np.ndarray, draft: int) -> tuple[bool, int | None]:
        if temperature <= 0:
            best = int(np.argmax(row))
            # None on success: a fully accepted run must take its next token from the row after
            # the last draft, not reuse the last draft's own logits.
            return draft == best, (None if draft == best else best)
        ids, probs = distribution(row / np.float32(temperature), top_p, top_k)
        hit = ids == draft
        if rng.random() < float(probs[hit].sum()):
            return True, None
        rest = ~hit
        if not rest.any():
            return True, None
        w = probs[rest]
        return False, int(rng.choice(ids[rest], p=w / w.sum()))

    return check


def speculative_generate(
    *,
    verify,
    commit,
    propose,
    first_logits: np.ndarray,
    max_new_tokens: int,
    room,
    max_drafts: int,
    pass_len: int = 0,
    step=None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    stop=(),
    rng: np.random.Generator | None = None,
    on_token=None,
    stats: dict | None = None,
) -> list[int]:
    """Speculative decode loop, independent of the engine.

    `verify(tokens) -> rows` runs one multi-token pass and returns one logits row per fed position
    (`rows[i]` predicts the token after `tokens[i]`); it may feed more tokens than it returns, but
    must not commit them. `commit(n)` keeps the first `n` fed tokens and discards the rest.
    `propose(sampled) -> drafts` returns at most `max_drafts` tokens. `room()` is the number of
    context positions left. A verify pass always feeds `pass_len` tokens (the graph's fixed shape),
    so when fewer positions remain and `step` is given, the tail decodes one token at a time
    instead of stopping short of the context end.
    """
    stop = set(stop)
    rng = rng if rng is not None else np.random.default_rng()
    check = acceptance(temperature, top_p, top_k, rng)
    if stats is None:
        stats = {}
    stats.update(drafted=0, accepted=0, passes=0)

    out: list[int] = []

    def emit(token: int) -> None:
        out.append(token)
        if on_token is not None:
            on_token(token)

    pending = draw(first_logits, temperature, top_p, top_k, rng)
    if max_new_tokens <= 0 or pending in stop:
        return out
    emit(pending)

    while len(out) < max_new_tokens:
        slots = room()
        if slots < 2:
            break
        if pass_len and slots < pass_len:
            if step is None:
                break
            token = draw(step(pending), temperature, top_p, top_k, rng)
            if token in stop:
                break
            emit(token)
            pending = token
            continue
        drafts = list(propose(pending))[: min(max_drafts, slots - 1)]
        if not drafts and step is not None:
            # Nothing to verify: a 17-token pass would cost ~1.4 decode steps and accept nothing.
            token = draw(step(pending), temperature, top_p, top_k, rng)
            if token in stop:
                break
            emit(token)
            pending = token
            continue
        rows = verify([pending] + drafts)
        stats["passes"] += 1
        stats["drafted"] += len(drafts)

        accepted = 0
        corrected: int | None = None
        while accepted < len(drafts):
            ok, corrected = check(rows[accepted], drafts[accepted])
            if not ok:
                break
            accepted += 1
        stats["accepted"] += accepted

        # Commit only what is emitted: the cache must stay consistent with the returned tokens,
        # so a draft burst cut by `max_new_tokens` (or by a stop token) is not left in the state.
        emitted = 0
        for token in drafts[: min(accepted, max_new_tokens - len(out))]:
            if token in stop:
                break
            emit(token)
            emitted += 1
        commit(1 + emitted)
        if emitted < accepted or len(out) >= max_new_tokens:
            break

        token = corrected if corrected is not None else draw(
            rows[len(drafts)], temperature, top_p, top_k, rng
        )
        if token in stop:
            break
        emit(token)
        pending = token
    return out
