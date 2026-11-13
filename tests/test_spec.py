"""Speculative decoding: proposer parity with llama.cpp, and exactness of the accept rule.

The loop is tested against a toy model, not Core ML: `speculative_generate` is engine-agnostic, and
a deterministic 4-token chain has logits we can compute by hand. The point of these tests is that
speculation must not change the output distribution — greedy speculation has to reproduce greedy
decode token for token, and sampled speculation has to reproduce the target's probabilities.
"""

import numpy as np
import pytest

from lone.spec import NgramProposer, distribution, draw, speculative_generate

# ---------------------------------------------------------------- proposer (llama.cpp parity)


def test_proposer_needs_n_plus_m_plus_one_tokens():
    # llama.cpp: `cur_len <= size_n + size_m + 1` returns no draft, even with a perfect repeat.
    assert NgramProposer(2, 2).propose([1, 2, 1, 2], 1) == []
    assert NgramProposer(2, 2).propose([1, 2, 1, 2, 1], 1) == []
    assert NgramProposer(2, 2).propose([1, 2, 7, 8, 1, 2, 3, 4, 1], 2) == [3, 4]


def test_proposer_copies_the_m_gram_after_the_most_recent_match():
    # pattern = last n-1 tokens + sampled = (1, 2, 3), matched at index 2; the m-gram after it
    # runs to the end of the history.
    history = [9, 1, 2, 3, 4, 5, 6, 7, 9, 9, 9, 1, 2]
    assert NgramProposer(3, 4).propose(history, 3) == [4, 5, 6, 7]


def test_proposer_ignores_a_match_at_position_zero():
    history = [1, 2, 3, 7, 8, 9, 10, 1, 2]
    assert NgramProposer(3, 4).propose(history, 3) == []


def test_proposer_drops_a_draft_shorter_than_n():
    # match at index 7 leaves copy_max = 3 < n == 4, so nothing is drafted.
    history = [0, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8, 5, 6, 7]
    assert NgramProposer(4, 8).propose(history, 8) == []


def test_proposer_truncates_at_the_end_of_history():
    # match at index 8 leaves 4 tokens after the n-gram; m is 8, so the draft stops at the end.
    history = [0] * 8 + [7, 8, 9, 10, 0, 7, 8, 9]
    assert NgramProposer(4, 8).propose(history, 10) == [0, 7, 8, 9]


def test_proposer_takes_the_most_recent_earlier_match():
    history = [1, 2, 3, 40, 41, 1, 2, 3, 50, 51, 1, 2]
    # pattern = (1, 2, 3); the later copy at index 5 wins over the one at index 0.
    assert NgramProposer(3, 4).propose(history, 3) == [50, 51, 1, 2]


# ------------------------------------------------------------------ distribution / draw


def test_distribution_is_the_vocabulary_wide_top_p_set():
    logits = np.array([5.0, 4.0, 1.0, 0.0, -3.0])
    ids, probs = distribution(logits.astype(np.float32), top_p=0.99)
    assert ids.tolist() == [0, 1]
    assert probs.sum() == pytest.approx(1.0)
    # P(0) = e^5 / (e^5 + e^4); P(1) = e^4 / (...), renormalised after top-p keeps both.
    p0 = np.exp(5.0) / (np.exp(5.0) + np.exp(4.0))
    assert probs[0] == pytest.approx(p0)


def test_draw_greedy_is_argmax():
    logits = np.array([0.0, 3.0, 1.0], dtype=np.float32)
    assert draw(logits, temperature=0.0) == 1


# --------------------------------------------------------------------- the loop, toy model
class ChainModel:
    """A deterministic 4-token model: logits are a function of the last token only.

    `verify` mimics the runtime contract — feed a multi-token pass, return one row per fed
    position, and only commit the prefix the loop asks for.
    """

    def __init__(self, table, start=0, ctx=4096):
        self.table = np.asarray(table, dtype=np.float32)  # table[last_token] -> logits
        self.tokens = [start]
        self.pos = start
        self.ctx = ctx
        self.passes = 0

    def room(self):
        return self.ctx - self.pos

    def verify(self, ids):
        self.passes += 1
        rows = np.stack([self.table[t] for t in ids])
        self._pass_start = self.pos
        self._fed = list(ids)
        return rows

    def commit(self, n):
        self.tokens.extend(self._fed[:n])
        self.pos = self._pass_start + n

    def step(self, token):
        """Single-token decode: the tail path when a full verify pass does not fit."""
        self.tokens.append(token)
        self.pos += 1
        return self.table[token]


def greedy_reference(model_factory, first_token, steps):
    """Plain greedy decode over the same chain."""
    model = model_factory()
    out = [first_token]
    for _ in range(steps):
        out.append(int(np.argmax(model.table[out[-1]])))
    return out


def test_greedy_speculation_reproduces_plain_greedy_decode():
    # Each token's logits prefer one successor; a proposer that drafts the same successor is
    # always right, and the loop must accept every draft.
    table = np.array([[0.0, 9.0, 0.0, 0.0], [0.0, 0.0, 9.0, 0.0],
                      [0.0, 0.0, 0.0, 9.0], [9.0, 0.0, 0.0, 0.0]])
    reference = greedy_reference(lambda: ChainModel(table), 0, 40)
    model = ChainModel(table)
    stats = {}
    out = speculative_generate(
        verify=model.verify, commit=model.commit,
        propose=lambda sampled: [int(np.argmax(table[sampled]))] * 8,
        first_logits=table[0], max_new_tokens=40, room=model.room, max_drafts=8,
        stats=stats,
    )
    assert out == reference[1:]
    assert stats["accepted"] > 0 and stats["passes"] > 0
    # The state must be a prefix of the emitted sequence: the last token may still be pending.
    assert ([0] + out)[: len(model.tokens)] == model.tokens
    assert len(model.tokens) >= len(out) - 1


def test_a_fully_accepted_run_takes_the_bonus_token_from_the_next_row():
    # 0 -> 1 -> 2 -> 3 -> 0; the proposer is always right, so every run is fully accepted and the
    # next token must come from the row after the last draft, not from the last draft's logits.
    table = np.array([[0.0, 9.0, 0.0, 0.0], [0.0, 0.0, 9.0, 0.0],
                      [0.0, 0.0, 0.0, 9.0], [9.0, 0.0, 0.0, 0.0]])
    model = ChainModel(table)

    def propose(sampled):
        nxt = int(np.argmax(table[sampled]))
        after = int(np.argmax(table[nxt]))
        return [nxt, after]

    out = speculative_generate(
        verify=model.verify, commit=model.commit, propose=propose,
        first_logits=table[0], max_new_tokens=12, room=model.room, max_drafts=2,
    )
    assert out == greedy_reference(lambda: ChainModel(table), 0, 12)[1:]


def test_a_wrong_draft_is_corrected_and_the_cache_rewound():
    # 0 -> 1 -> 2 -> 3 -> 0; the proposer lies about the token after 1.
    table = np.array([[0.0, 9.0, 0.0, 0.0], [0.0, 0.0, 9.0, 0.0],
                      [0.0, 0.0, 0.0, 9.0], [9.0, 0.0, 0.0, 0.0]])
    model = ChainModel(table)

    def propose(sampled):
        return [3, 3, 3] if sampled == 1 else [int(np.argmax(table[sampled]))] * 3

    out = speculative_generate(
        verify=model.verify, commit=model.commit, propose=propose,
        first_logits=table[0], max_new_tokens=12, room=model.room, max_drafts=3,
    )
    assert out == greedy_reference(lambda: ChainModel(table), 0, 12)[1:]
    assert ([0] + out)[: len(model.tokens)] == model.tokens  # no rejected draft survived


def test_context_tail_falls_back_to_single_token_decode():
    # A verify pass feeds pass_len tokens; when the context has less room than that, the loop must
    # decode one token at a time rather than stop short of the end.
    table = np.array([[0.0, 9.0, 0.0, 0.0], [0.0, 0.0, 9.0, 0.0],
                      [0.0, 0.0, 0.0, 9.0], [9.0, 0.0, 0.0, 0.0]])
    model = ChainModel(table, ctx=9)
    out = speculative_generate(
        verify=model.verify, commit=model.commit, propose=lambda s: [],
        first_logits=table[0], max_new_tokens=64, room=model.room, max_drafts=3,
        pass_len=4, step=model.step,
    )
    assert len(out) == 9  # ctx 9 from position 0: the same count plain greedy decode produces
    assert out == greedy_reference(lambda: ChainModel(table), 0, 9)[1:]
    assert model.pos == 8  # the last emitted token is pending, not yet fed


def test_sampled_speculation_matches_the_target_distribution():
    """20k draws through the spec loop must match the analytic distribution of the target.

    The proposer always drafts token 0, so most drafts are rejected and the residual path does the
    work — the case where a wrong accept rule would skew the output.
    """
    logits = np.array([1.6, 1.1, 0.2, -0.7], dtype=np.float32)
    table = np.tile(logits, (4, 1))  # every state has the same distribution
    ids, probs = distribution(logits / np.float32(0.7), 1.0, 0)
    p = np.zeros(4)
    p[ids] = probs

    model = ChainModel(table)
    out = speculative_generate(
        verify=model.verify, commit=model.commit, propose=lambda s: [0, 0],
        first_logits=logits, max_new_tokens=20_000, room=model.room, max_drafts=2,
        temperature=0.7, rng=np.random.default_rng(1234),
    )
    counts = np.bincount(out[1:], minlength=4) / (len(out) - 1)
    assert np.allclose(counts, p, atol=0.015), (counts, p)


def test_an_empty_draft_takes_the_decode_path_not_a_verify_pass():
    # A verify pass costs more than a decode step; with nothing to verify, skip it.
    table = np.array([[0.0, 9.0, 0.0, 0.0], [0.0, 0.0, 9.0, 0.0],
                      [0.0, 0.0, 0.0, 9.0], [9.0, 0.0, 0.0, 0.0]])
    model = ChainModel(table)
    out = speculative_generate(
        verify=model.verify, commit=model.commit, propose=lambda s: [],
        first_logits=table[0], max_new_tokens=8, room=model.room, max_drafts=3,
        pass_len=4, step=model.step,
    )
    assert out == greedy_reference(lambda: ChainModel(table), 0, 8)[1:]
    assert model.passes == 0  # no verify pass was issued


def test_context_without_a_fallback_stops_before_the_overflow():
    table = np.eye(4, dtype=np.float32) * 9.0
    model = ChainModel(table, ctx=10)
    out = speculative_generate(
        verify=model.verify, commit=model.commit, propose=lambda s: [s] * 8,
        first_logits=table[0], max_new_tokens=64, room=model.room, max_drafts=8, pass_len=5,
    )
    assert len(out) <= 10


def test_max_new_tokens_is_not_overshot_by_a_draft_burst():
    table = np.array([[0.0, 9.0, 0.0, 0.0], [0.0, 0.0, 9.0, 0.0],
                      [0.0, 0.0, 0.0, 9.0], [9.0, 0.0, 0.0, 0.0]])
    model = ChainModel(table)
    out = speculative_generate(
        verify=model.verify, commit=model.commit, propose=lambda s: [int(np.argmax(table[s]))] * 8,
        first_logits=table[0], max_new_tokens=5, room=model.room, max_drafts=8,
    )
    assert len(out) == 5
    assert ([0] + out)[: len(model.tokens)] == model.tokens
