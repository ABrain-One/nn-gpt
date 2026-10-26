"""The generation engine must not hold the GPU while the networks are evaluated.

Every generated network is trained on the same card the engine is running on.
sleep() parks the engine but keeps its CUDA context, its compiled graphs and
the pool reserved through gpu_memory_utilization, so the evaluator ran against
whatever was left over. The engine is therefore destroyed when an epoch's
generation finishes and rebuilt when the next epoch needs it.

Loaded with torch and the vllm distributed hooks stubbed, so the lifecycle is
checked on any machine rather than only on one with a GPU.
"""
import importlib.util
import sys
import types
from pathlib import Path

from _unittest_bridge import load_tests_from, pytest  # stdlib stand-in without pytest

ROOT = Path(__file__).resolve().parents[1]


def _stub_torch():
    torch = types.ModuleType("torch")
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
        synchronize=lambda: None,
        mem_get_info=lambda: (8 * 1024 ** 3, 24 * 1024 ** 3),
        current_device=lambda: 0,
    )
    torch.cuda = cuda
    return torch


def _load(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _stub_torch())
    util = types.ModuleType("ab.gpt.util.Util")
    util.extract_all_to_train = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "ab.gpt.util.Util", util)
    spec = importlib.util.spec_from_file_location(
        "_vllm_sleep_under_test", ROOT / "ab/gpt/util/llm/VLLMSleep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeCore:
    def __init__(self):
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1


class _FakeLLM:
    def __init__(self):
        self.llm_engine = types.SimpleNamespace(engine_core=_FakeCore())


class _FakeEngine:
    """Stands in for VLLMSleepEngine with the real shutdown()/sleep() contract."""

    def __init__(self, V):
        self.V = V
        self.llm = _FakeLLM()
        self._asleep = False
        self.adapter = None
        self.destroyed = False
        self.sleeps = 0

    def sleep(self, level=1):
        self.sleeps += 1
        self._asleep = True

    def wake_up(self, adapter_path=None):
        self._asleep = False
        self.adapter = adapter_path

    def set_adapter(self, adapter_path):
        self.adapter = adapter_path

    def shutdown(self):
        self.V.VLLMSleepEngine.shutdown(self)
        self.destroyed = True

    def token_length(self, text):
        return len(text)


@pytest.fixture
def V(monkeypatch):
    return _load(monkeypatch)


def _generation(V, monkeypatch, built):
    monkeypatch.setattr(V, "park_trainer", lambda model: "cuda:0")
    monkeypatch.setattr(V, "unpark_trainer", lambda model, dev: None)

    def _build(*a, **k):
        eng = _FakeEngine(V)
        built.append(eng)
        return eng

    monkeypatch.setattr(V, "build_engine", _build)
    gen = V.VLLMGeneration("m", _tokenizer(), 8192)
    return gen


class _tokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return list(range(sum(len(m["content"]) for m in messages)))

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def test_release_destroys_the_engine_and_frees_the_card(V, monkeypatch):
    built = []
    gen = _generation(V, monkeypatch, built)

    gen.prepare(None, object())
    assert gen.active and len(built) == 1

    gen.release(object())

    assert built[0].destroyed, "the engine survived the end of generation"
    assert built[0].llm is None, "the engine still holds its LLM"
    assert gen.engine is None and not gen.active


def test_the_next_epoch_rebuilds_rather_than_generating_with_nothing(V, monkeypatch, tmp_path):
    built = []
    gen = _generation(V, monkeypatch, built)

    gen.prepare(None, object())
    gen.release(object())
    adapter = tmp_path / "A0"
    gen.prepare(adapter, object())

    assert len(built) == 2, "the engine was not rebuilt for the new epoch"
    assert gen.active
    assert built[1].adapter == adapter, "the rebuilt engine lost the adapter"


def test_the_engine_core_is_shut_down_not_merely_dereferenced(V, monkeypatch):
    built = []
    gen = _generation(V, monkeypatch, built)
    gen.prepare(None, object())
    core = built[0].llm.llm_engine.engine_core

    gen.release(object())

    assert core.shutdowns == 1, "the engine core was left running"


def test_the_model_parallel_group_is_unregistered_so_a_rebuild_can_work(V, monkeypatch):
    """A registered group makes the next build fail outright."""
    calls = []
    parallel = types.ModuleType("vllm.distributed.parallel_state")
    parallel.destroy_model_parallel = lambda: calls.append("model")
    parallel.destroy_distributed_environment = lambda: calls.append("env")
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.distributed",
                        types.ModuleType("vllm.distributed"))
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel)

    built = []
    gen = _generation(V, monkeypatch, built)
    gen.prepare(None, object())
    gen.release(object())

    assert calls == ["model", "env"]


def test_teardown_can_be_turned_off_in_favour_of_sleep(V, monkeypatch):
    monkeypatch.setattr(V, "TEARDOWN_BETWEEN_EPOCHS", False)
    built = []
    gen = _generation(V, monkeypatch, built)

    gen.prepare(None, object())
    gen.release(object())

    assert not built[0].destroyed
    assert built[0].sleeps == 1
    assert gen.engine is built[0], "the engine was dropped despite teardown being off"


def test_the_trainer_comes_back_to_the_gpu_either_way(V, monkeypatch):
    unparked = []
    built = []
    gen = _generation(V, monkeypatch, built)
    monkeypatch.setattr(V, "unpark_trainer", lambda model, dev: unparked.append(dev))

    model = object()
    gen.prepare(None, model)
    gen.release(model)

    assert unparked == ["cuda:0"]


def test_prompts_can_still_be_measured_while_the_engine_is_gone(V, monkeypatch):
    """The prompt-fitting pass runs between epochs; None would skip every prompt."""
    built = []
    gen = _generation(V, monkeypatch, built)
    gen.prepare(None, object())
    gen.release(object())

    assert gen.engine is None and gen.fallback is None
    n = gen.token_length("some prompt")
    assert isinstance(n, int) and n > 0
    assert gen.output_budget(n, 4096) == 4096


def test_ensure_asleep_is_harmless_once_the_engine_is_destroyed(V, monkeypatch):
    built = []
    gen = _generation(V, monkeypatch, built)
    gen.prepare(None, object())
    gen.release(object())

    gen.ensure_asleep()          # fine-tuning calls this every epoch
    gen.shutdown()               # and the run calls this at the end
    assert gen.engine is None


def test_shutdown_survives_an_engine_that_refuses_to_close(V, monkeypatch):
    """A failure to tear down must not take the run with it."""
    class _Stubborn:
        def shutdown(self):
            raise RuntimeError("nope")

    eng = _FakeEngine(V)
    eng.llm.llm_engine.engine_core = _Stubborn()
    V.VLLMSleepEngine.shutdown(eng)
    assert eng.llm is None


load_tests = load_tests_from(globals())
