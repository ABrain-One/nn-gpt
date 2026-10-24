"""Loop control for the generate -> evaluate -> fine-tune cycle.

Regression: the run ended immediately after the last fine-tune, so the final
adapter was never generated from, never validated, and never recorded. A
closing validation pass now runs once before the graph terminates.

The wiring mirrors ab/gpt/act/agents/run_agent.py:
    entry     -> manager
    manager   -> conditional on next_action
    generator -> conditional on next_action   (not back through manager)
    evaluator -> manager
    finetuner -> manager
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

from _unittest_bridge import load_tests_from, pytest  # stdlib stand-in without pytest

ROOT = Path(__file__).resolve().parents[1]


def _manager():
    stub = type(sys)("stub")
    stub.AgentState = dict
    sys.modules.setdefault("ab.gpt.act.agents.state", stub)
    spec = importlib.util.spec_from_file_location(
        "_manager_under_test", ROOT / "ab/gpt/act/agents/manager.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MGR = _manager()


class _State(dict):
    def get(self, key, default=None):
        value = dict.get(self, key, None)
        return value if value is not None else default


def _run(total, generation_fails_at=()):
    st = _State(current_epoch=0, llm_tune_epochs=total, next_action="generate")
    node, trace, guard = "manager", [], 0
    while node != "END":
        guard += 1
        assert guard < 200, "loop did not terminate: " + " -> ".join(trace)
        if node == "manager":
            with contextlib.redirect_stdout(io.StringIO()):
                st.update(MGR.manager_node(st))
            node = {"generate": "generator", "evaluate": "evaluator",
                    "finetune": "finetuner", "end": "END"}[st["next_action"]]
        elif node == "generator":
            trace.append(f"e{st['current_epoch']}:gen*" if st.get("final_eval")
                         else f"e{st['current_epoch']}:gen")
            if st["current_epoch"] in generation_fails_at:
                trace[-1] += "(no code)"
                if st.get("final_eval"):
                    st.update(next_action="end", final_eval_done=True)
                else:
                    st["next_action"] = "finetune"
            else:
                st["next_action"] = "evaluate"
            node = {"evaluate": "evaluator", "finetune": "finetuner",
                    "end": "END"}[st["next_action"]]
        elif node == "evaluator":
            trace.append(f"e{st['current_epoch']}:eval")
            if st.get("final_eval"):
                st.update(next_action="end", final_eval_done=True)
            else:
                st["next_action"] = "finetune"
            node = "manager"
        elif node == "finetuner":
            trace.append(f"e{st['current_epoch']}:ft")
            st.update(current_epoch=st["current_epoch"] + 1, next_action="generate")
            node = "manager"
    return trace


@pytest.mark.parametrize("total", [1, 2, 4])
def test_every_finetune_is_followed_by_a_validation(total):
    trace = _run(total)
    assert len([x for x in trace if x.endswith(":ft")]) == total
    # baseline (before any fine-tune) plus one validation per fine-tune
    assert len([x for x in trace if x.endswith(":eval")]) == total + 1
    assert trace[-1] == f"e{total}:eval"


def test_closing_pass_is_marked_and_runs_once():
    trace = _run(4)
    closing = [x for x in trace if x.endswith(":gen*")]
    assert closing == ["e4:gen*"]


def test_closing_pass_terminates_when_it_generates_nothing():
    trace = _run(4, generation_fails_at={4})
    assert trace[-1] == "e4:gen*(no code)"


def test_midrun_collapse_does_not_abort_the_run():
    trace = _run(4, generation_fails_at={2})
    assert len([x for x in trace if x.endswith(":ft")]) == 4
    assert trace[-1] == "e4:eval"


@pytest.mark.parametrize("total", [1, 2, 4])
def test_closing_pass_can_be_skipped(total, monkeypatch):
    monkeypatch.setenv("NNGPT_SKIP_CLOSING_EVAL", "1")
    trace = _run(total)
    assert not [x for x in trace if x.endswith(":gen*")]
    assert len([x for x in trace if x.endswith(":ft")]) == total
    assert len([x for x in trace if x.endswith(":eval")]) == total
    assert trace[-1] == f"e{total - 1}:ft"


load_tests = load_tests_from(globals())
