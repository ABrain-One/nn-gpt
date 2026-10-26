"""With default settings (none of the new environment flags set) every entry point behaves as before.

Covers every launcher under ab/gpt/act that calls ab.gpt.act.tune.Tune.main: its sampling settings reach vLLM's
SamplingParams unchanged, the iterative child gets no new arguments, --peft without carry-over only warns, and a
whole-number --warmup_steps keeps its value. Launchers are read with ast (not imported), so no model is needed.
"""
import ast
import importlib.machinery
import sys
import types
from pathlib import Path

from _unittest_bridge import load_tests_from, pytest  # stdlib stand-in without pytest

ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "ab" / "gpt" / "act"
TUNE_CLI = ACT / "tune" / "Tune.py"
FINETUNE = ACT / "tune" / "iterative" / "finetune.py"
UTIL_TUNE = ROOT / "ab" / "gpt" / "util" / "tune" / "Tune.py"
FLAGS = ("NNGPT_CARRY_ADAPTER", "NNGPT_VLLM_INHERIT_GEN_CONFIG", "NNGPT_VLLM_REPETITION_PENALTY",
         "NNGPT_SKIP_CLOSING_EVAL")


def _function(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _constants(path):
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


def _launchers():
    found = []
    for path in sorted(ACT.rglob("*.py")):
        text = path.read_text(errors="replace")
        if "TuneNNGen.main(" not in text or "def main" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "main"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "TuneNNGen"):
                kwargs = {}
                for kw in node.keywords:
                    try:
                        kwargs[kw.arg] = ast.literal_eval(kw.value)
                    except ValueError:
                        kwargs[kw.arg] = "<expr>"
                found.append((str(path.relative_to(ROOT)), kwargs))
    return found


LAUNCHERS = _launchers()
DEFAULTS = _constants(TUNE_CLI)


@pytest.fixture(autouse=True)
def _no_new_flags(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def engine_class(monkeypatch, tmp_path):
    """The real VLLMSleepEngine with a stub vllm package that records SamplingParams."""
    from ab.gpt.util.llm import VLLMSleep
    # What a machine without CUDA reports. The sampling parameters are the
    # subject here, and probing a real card makes the test fail whenever
    # something else is using it.
    monkeypatch.setattr(VLLMSleep, "_free_gpu", lambda: (0, 0))
    calls = []
    vllm = types.ModuleType("vllm")
    vllm.SamplingParams = lambda **kw: calls.append(kw) or kw

    class _LLM:
        def __init__(self, **kw):
            pass

        def generate(self, requests, sampling, lora_request=None):
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="")]) for _ in requests]

    vllm.LLM = _LLM
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: prompt_token_ids
    for name, mod in {"vllm": vllm, "vllm.inputs": inputs}.items():
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        monkeypatch.setitem(sys.modules, name, mod)
    (tmp_path / "generation_config.json").write_text('{"repetition_penalty": 1.1, "temperature": 0.7}')

    class _Tok:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            return [1, 2, 3]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2, 3]}

    def build(**kw):
        engine = VLLMSleep.VLLMSleepEngine(str(tmp_path), _Tok(), 8192, **kw)
        engine.chat_batch(["prompt"], max_new_tokens=2048)
        return calls[-1]
    return build


def test_launchers_found():
    names = [p for p, _ in LAUNCHERS]
    assert any(p.endswith("qwen2_5_14B_channel_alter.py") for p in names)
    assert any(p.endswith("qwen2_5_32B_channel_alter.py") for p in names)


@pytest.mark.parametrize("path,kwargs", LAUNCHERS, ids=[p for p, _ in LAUNCHERS])
def test_sampling_params_unchanged_by_default(engine_class, path, kwargs):
    t = kwargs.get("temperature", DEFAULTS["TEMPERATURE"])
    k = kwargs.get("top_k", DEFAULTS["TOP_K"])
    p = kwargs.get("top_p", DEFAULTS["TOP_P"])
    if "<expr>" in (t, k, p):
        pytest.skip("sampling settings computed at run time")
    assert engine_class(temperature=t, top_k=k, top_p=p) == {
        "temperature": t, "top_k": k, "top_p": p, "max_tokens": 2048}


@pytest.mark.parametrize("path,kwargs", LAUNCHERS, ids=[p for p, _ in LAUNCHERS])
def test_iterative_child_gets_no_new_arguments_by_default(path, kwargs):
    flags = _function(FINETUNE, "_vllm_child_flags")
    for cycle in (1, 2, 3):
        args, _ = flags(cycle, {})
        assert args == []


@pytest.mark.parametrize("path,kwargs", LAUNCHERS, ids=[p for p, _ in LAUNCHERS])
def test_peft_without_carry_only_warns(path, kwargs):
    warn = _function(UTIL_TUNE, "_vllm_peft_warning")
    peft = kwargs.get("peft", DEFAULTS.get("PEFT"))
    message = warn(object(), peft, None)  # vLLM engine present, no carried adapter
    assert (message is None) == (not peft)


@pytest.mark.parametrize("path,kwargs", LAUNCHERS, ids=[p for p, _ in LAUNCHERS])
def test_warmup_steps_value_kept(path, kwargs):
    parse = _function(TUNE_CLI, "_warmup_steps_arg")
    value = kwargs.get("warmup_steps", DEFAULTS["WARMUP_STEPS"])
    if value == "<expr>":
        pytest.skip("computed at run time")
    parsed = parse(str(value))
    assert parsed == value
    if float(value).is_integer():
        assert isinstance(parsed, int)


def test_flags_enable_the_fixes(engine_class, monkeypatch):
    flags = _function(FINETUNE, "_vllm_child_flags")
    assert flags(2, {"NNGPT_CARRY_ADAPTER": "1"})[0] == ["--use_vllm"]
    assert flags(2, {"NNGPT_CARRY_ADAPTER": "1", "NNGPT_VLLM_SLEEP": "0"})[0] == ["--no-use_vllm"]
    assert flags(2, {})[1] is not None  # vLLM without carry-over: a warning, no new argument
    monkeypatch.setenv("NNGPT_VLLM_INHERIT_GEN_CONFIG", "1")
    assert engine_class(temperature=0.2, top_k=50, top_p=0.9)["repetition_penalty"] == 1.1


load_tests = load_tests_from(globals())
