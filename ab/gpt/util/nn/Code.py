import ast
import io
import tokenize


def dedup_imports(src: str) -> str:
    """Merge duplicate torch import lines."""
    try:
        seen, out = set(), []
        for ln in src.splitlines():
            if ln.startswith("import torch") or ln.startswith("from torch import nn"):
                if ln in seen:
                    continue
                seen.add(ln)
            out.append(ln)
        return "\n".join(out)
    except:
        return src


# ────────────────────────────────────────────────────────────────
#  Normalise stray top-level indentation (safe for half-finished files)
# ────────────────────────────────────────────────────────────────
def normalize_top_indent(src: str) -> str:
    try:
        ast.parse(src)
        return src
    except (SyntaxError, IndentationError):
        pass
    try:
        lines = src.splitlines(True)
        for i, ln in enumerate(lines):
            if ln.strip():
                lines[i] = ln.lstrip()
                break
        return "".join(lines)
    except:
        return src


def remove_inline_comments(code: str) -> str:
    """
    Remove inline comments from code using tokenization.
    This function ONLY removes COMMENT tokens, not docstrings.
    Safe to use because it doesn't skip STRING tokens.
    """
    # ------------------------------------------------------------
    # io.StringIO(code)：
    #   Wraps a string so it behaves like a file. 
    #   Provides a file-like interface to a string.
    # .readline: 
    #   A method that returns the next line each time it’s called. 
    #   tokenize.generate_tokens() calls it repeatedly to read lines.
    # generate_tokens: 
    #   reads line by line and produces tokens.
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Example:
    #   x = 5  # This is a comment
    #   y = x + 10  # This is a comment
    #   NAME('x'), OP('='), NUMBER('5'), COMMENT('# This is a comment'), NEWLINE, 
    #   NAME('y'), OP('='), NAME('x'), OP('+'), NUMBER('10'), COMMENT('# This is a comment'), NEWLINE,
    # ------------------------------------------------------------
    try:
        result = []
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)

        for tok in tokens:
            tok_type, tok_string, _, _, _ = tok

            # Skip ONLY comment tokens
            if tok_type == tokenize.COMMENT:
                continue
            # Keep everything else (including STRING tokens)
            result.append(tok)

        return tokenize.untokenize(result).strip()
    except:
        # If tokenization fails, return original code
        return code


def strip_comments(code: str) -> str:
    """
    Remove both docstrings and comments from Python code.
    Uses a two-step approach:
    1. AST-based docstring removal (clean, no backslashes)
    2. Tokenization-based comment removal (safe, only removes comments)
    """
    try:
        result = []

        # STEP 1: Parse the code into an Abstract Syntax Tree
        tree = ast.parse(code)

        # Remove docstrings from modules, classes, and functions
        for node in ast.walk(tree):  # walk every single node in the AST
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if node.body and isinstance(node.body[0], ast.Expr):
                    # body: a list of all statements inside a function/class/module.
                    # Check if the first statement is a string constant (docstring)

                    # def foo(x):                    # node = FunctionDef
                    #     """Docstring"""            # node.body[0] = Expr (the docstring)
                    #     y = x + 1                  # node.body[1] = Assign
                    #     return y                   # node.body[2] = Return
                    expr_value = node.body[0].value
                    if isinstance(expr_value, ast.Constant) and isinstance(expr_value.value, str):
                        # Remove the docstring
                        node.body.pop(0)
                    elif isinstance(expr_value, ast.Str):  # For Python < 3.8 compatibility
                        node.body.pop(0)

        # Convert the AST back to source code
        code_without_docstrings = ast.unparse(tree)

        # STEP 2: Remove inline comments from the code
        # Note: ast.unparse() preserves comments, so we need this extra step
        code_without_comments = remove_inline_comments(code_without_docstrings)

        return code_without_comments
    except:
        return code


def improve_code(code: str) -> str:
    try:
        return normalize_top_indent(dedup_imports(strip_comments(code)))
    except:
        pass
    return code
