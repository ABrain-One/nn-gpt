"""generate_step must hand the vLLM engine the carried-over adapter at epoch 0.

The engine loads the base weights by name and applies an adapter from disk; with None at epoch 0 every
cycle of the iterative pipeline (one epoch per child process) generated with the base model."""
try:
    from ab.gpt.util.tune import Tune as tune_mod
except ImportError as exc:
    from _unittest_bridge import skip_module

    skip_module(f"test_vllm_epoch0_adapter needs ab.gpt (and so torch): {exc}")


class _Stop(Exception):
    pass


class _RecordingEngine:
    fallback = None
    adapter = "unset"

    def prepare(self, adapter, model):
        self.adapter = adapter
        raise _Stop  # generation itself is not under test


def _prepared_adapter(epoch, init_adapter):
    engine = _RecordingEngine()
    state = {"current_epoch": epoch, "skip_epoch": 0, "vllm_engine": engine, "chat_bot_hf": None,
             "model": None, "base_model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
             "vllm_init_adapter": init_adapter}
    try:
        tune_mod.generate_step(state)
    except _Stop:
        pass
    return engine.adapter


def test_epoch0_uses_carried_adapter():
    assert _prepared_adapter(0, "/runs/r/cycle_1/checkpoint") == "/runs/r/cycle_1/checkpoint"


def test_epoch0_without_carry_uses_base():
    assert _prepared_adapter(0, None) is None


def test_later_epoch_uses_previous_epoch_adapter():
    adapter = _prepared_adapter(1, "/runs/r/cycle_1/checkpoint")
    assert str(adapter).endswith("Qwen/Qwen2.5-Coder-7B-Instruct")
    assert str(adapter) != "/runs/r/cycle_1/checkpoint"


from _unittest_bridge import load_tests_from  # noqa: E402

load_tests = load_tests_from(globals())
