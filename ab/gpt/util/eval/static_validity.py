"""
Static structural validity checker for generated new_nn.py files.

Does NOT execute any code — uses AST parsing only.
Decouples generation quality from infra crashes (stale tmp paths, LEMUR duplicates).

LEMUR NN structure (as in Eval.py lines 51-53):
  - Module-level function : supported_hyperparameters()   ← NOT a Net method
  - Net(nn.Module) methods: __init__, train_setup, learn, forward

Usage:
    from ab.gpt.util.eval.static_validity import check_static_validity
    result = check_static_validity("/path/to/new_nn.py")
    print(result.valid, result.reason, result.score)
"""

import ast
import os
from dataclasses import dataclass
from typing import Optional

REQUIRED_CLASS = "Net"
# supported_hyperparameters is a MODULE-LEVEL function, not a method of Net
REQUIRED_MODULE_FUNS = {"supported_hyperparameters"}
REQUIRED_NET_METHODS = {"__init__", "train_setup", "learn", "forward"}
MIN_CODE_LINES = 10
MIN_LEARN_STMTS = 1


@dataclass
class ValidityResult:
    valid: bool            # True = passes all structural checks
    score: float           # 0.0–1.0 partial-credit score for analysis
    reason: str            # human-readable verdict
    has_net_class: bool = False
    has_supported_hyperparameters: bool = False
    has_forward_or_learn: bool = False
    has_nontrivial_body: bool = False
    parse_error: Optional[str] = None
    nn_chars: int = 0
    nn_lines: int = 0


def check_static_validity(nn_file: str) -> ValidityResult:
    """
    Return a ValidityResult for the given new_nn.py path.
    Never raises; all failures are captured in the result.
    """
    if not os.path.isfile(nn_file):
        return ValidityResult(False, 0.0, "file_not_found")

    try:
        with open(nn_file, encoding="utf-8", errors="replace") as f:
            source = f.read()
    except Exception as e:
        return ValidityResult(False, 0.0, f"read_error: {e}")

    nn_chars = len(source)
    nn_lines = source.count("\n") + 1

    if nn_chars < 50:
        return ValidityResult(
            False, 0.0, "code_too_short",
            nn_chars=nn_chars, nn_lines=nn_lines,
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ValidityResult(
            False, 0.1, f"syntax_error: {e}",
            parse_error=str(e),
            nn_chars=nn_chars, nn_lines=nn_lines,
        )

    # Check module-level functions (supported_hyperparameters lives here, NOT in Net)
    module_funs = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    has_sph = "supported_hyperparameters" in module_funs

    # Find the Net class (search only top-level class definitions)
    net_class: Optional[ast.ClassDef] = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == REQUIRED_CLASS:
            net_class = node
            break

    has_net_class = net_class is not None
    if not has_net_class:
        score = 0.2 + (0.1 if has_sph else 0.0)
        return ValidityResult(
            False, score, "no_Net_class",
            has_net_class=False,
            has_supported_hyperparameters=has_sph,
            nn_chars=nn_chars, nn_lines=nn_lines,
        )

    # Collect direct method names inside Net
    method_names = {
        node.name for node in net_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    has_init = "__init__" in method_names
    has_train_setup = "train_setup" in method_names
    has_learn = "learn" in method_names
    has_fwd = "forward" in method_names

    # Non-trivial learn body: at least MIN_LEARN_STMTS real statements
    has_nontrivial = False
    for node in net_class.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "learn":
            body_stmts = len([n for n in node.body if not isinstance(n, (ast.Pass, ast.Expr))])
            if body_stmts >= MIN_LEARN_STMTS:
                has_nontrivial = True

    has_min_lines = nn_lines >= MIN_CODE_LINES

    # Score
    score = 0.0
    if has_sph:
        score += 0.15
    if has_net_class:
        score += 0.10
    if has_init:
        score += 0.10
    if has_train_setup:
        score += 0.20
    if has_learn:
        score += 0.20
    if has_fwd:
        score += 0.15
    if has_nontrivial:
        score += 0.05
    if has_min_lines:
        score += 0.05

    valid = (has_sph and has_net_class and has_init and
             has_train_setup and has_learn and has_fwd and
             has_nontrivial and has_min_lines)

    if valid:
        reason = "structurally_valid"
    elif not has_sph:
        reason = "missing_supported_hyperparameters"
    elif not has_init:
        reason = "missing___init__"
    elif not has_train_setup:
        reason = "missing_train_setup"
    elif not has_learn:
        reason = "missing_learn"
    elif not has_fwd:
        reason = "missing_forward"
    elif not has_nontrivial:
        reason = "trivial_learn_body"
    else:
        reason = "insufficient_lines"

    return ValidityResult(
        valid=valid,
        score=score,
        reason=reason,
        has_net_class=has_net_class,
        has_supported_hyperparameters=has_sph,
        has_forward_or_learn=has_fwd or has_learn,
        has_nontrivial_body=has_nontrivial,
        nn_chars=nn_chars,
        nn_lines=nn_lines,
    )


def batch_validity(batch_dir: str) -> dict:
    """Check validity for a batch directory; returns a JSON-safe dict."""
    nn_file = os.path.join(batch_dir, "new_nn.py")
    r = check_static_validity(nn_file)
    return {
        "valid": r.valid,
        "score": r.score,
        "reason": r.reason,
        "has_net_class": r.has_net_class,
        "has_supported_hyperparameters": r.has_supported_hyperparameters,
        "has_forward_or_learn": r.has_forward_or_learn,
        "has_nontrivial_body": r.has_nontrivial_body,
        "nn_chars": r.nn_chars,
        "nn_lines": r.nn_lines,
    }
