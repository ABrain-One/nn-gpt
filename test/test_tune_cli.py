"""The Tune command line must parse: every option is defined once (argparse raises on duplicates at start-up).

The module is read with ast, not imported, so the test needs neither torch nor the LEMUR database."""
import ast
from collections import Counter
from pathlib import Path

TUNE = Path(__file__).resolve().parents[1] / "ab" / "gpt" / "act" / "tune" / "Tune.py"


def _option_strings(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    yield arg.value


def test_every_cli_option_is_defined_once():
    counts = Counter(_option_strings(ast.parse(TUNE.read_text())))
    assert counts, "no add_argument calls found"
    duplicates = {opt: n for opt, n in counts.items() if n > 1}
    assert not duplicates, f"options defined more than once: {duplicates}"


def _load_function(name):
    """Compile one top-level function of Tune.py on its own (no module import)."""
    tree = ast.parse(TUNE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TUNE), "exec"), namespace)
    return namespace[name]


def test_warmup_steps_whole_numbers_stay_int():
    # The iterative pipeline passes "--warmup_steps 20"; transformers 4.x rejects 20.0.
    parse = _load_function("_warmup_steps_arg")
    assert parse("20") == 20 and isinstance(parse("20"), int)
    assert parse("20.0") == 20 and isinstance(parse("20.0"), int)
    assert parse("0.05") == 0.05 and isinstance(parse("0.05"), float)


from _unittest_bridge import load_tests_from  # noqa: E402

load_tests = load_tests_from(globals())
