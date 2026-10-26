"""Generation must still produce test_nn models when some prompts do not fit.

A prompt that leaves no room for the answer used to be skipped inside the
generation loop, after its slot had already been spent: the epoch asked for 16
models, sent 16 prompts, and got back fewer -- with an empty full_output.txt
wherever a prompt had been dropped. The seed pool is now over-provisioned and
the length filter runs before anything is sent, so a dropped prompt is replaced
rather than lost.
"""

from _unittest_bridge import load_tests_from, pytest  # stdlib stand-in without pytest

pytest.importorskip("torch")

from ab.gpt.util.tune.Tune import (          # noqa: E402
    _seed_pool_size, _select_fitting_prompts, SEED_POOL_FACTOR,
)

CTX = 4096
MAX_NEW = 2048


class _Bot:
    """Minimal stand-in exposing the engine's measurement interface."""

    def __init__(self, lengths):
        self.lengths = lengths

    def token_length(self, text):
        return self.lengths[text]

    def output_budget(self, in_len, max_new_tokens):
        room = CTX - in_len
        return None if room < 256 else min(max_new_tokens or CTX, room)


def _prompts(lengths):
    return [("sys", text, None, "code") for text in lengths]


def test_pool_is_larger_than_the_number_of_models_requested():
    assert _seed_pool_size(16) > 16
    assert _seed_pool_size(16) == max(int(16 * SEED_POOL_FACTOR), 24)
    assert _seed_pool_size(1) > 1


def test_oversized_prompts_are_replaced_not_subtracted():
    lengths = {f"p{i}": (9000 if i < 6 else 2000) for i in range(_seed_pool_size(16))}
    accepted, dropped = _select_fitting_prompts(
        _prompts(lengths), _Bot(lengths), 16, MAX_NEW, None)

    assert dropped == 6
    assert len(accepted) == 16, "a dropped prompt cost the epoch a model"
    assert all(lengths[p[1]] == 2000 for p in accepted)


def test_nothing_is_sent_when_no_prompt_fits():
    lengths = {f"p{i}": 9000 for i in range(10)}
    accepted, dropped = _select_fitting_prompts(
        _prompts(lengths), _Bot(lengths), 16, MAX_NEW, None)

    assert accepted == []
    assert dropped == 10


def test_the_pool_is_not_drained_past_what_was_asked_for():
    lengths = {f"p{i}": 2000 for i in range(40)}
    accepted, dropped = _select_fitting_prompts(
        _prompts(lengths), _Bot(lengths), 16, MAX_NEW, None)

    assert len(accepted) == 16
    assert dropped == 0


def test_the_unsloth_input_cap_is_still_honoured():
    lengths = {f"p{i}": 2000 for i in range(5)}
    accepted, dropped = _select_fitting_prompts(
        _prompts(lengths), _Bot(lengths), 5, MAX_NEW, 1500)

    assert accepted == []
    assert dropped == 5


load_tests = load_tests_from(globals())
