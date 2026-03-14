from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SKIP_DIRS = {".git", ".venv", "__pycache__"}
INCLUDE_DIRS = {
    "nn-gpt": ["tools", "nn", "util", "db", "ab"],
    "nn-dataset": ["ab", "nn", "util", "db"],
}
MAX_FILE_SIZE = 2 * 1024 * 1024
PROGRESS_EVERY = 200
CALL_NAMES = {"execute", "executemany", "executescript"}
SQL_KEYWORDS = {
    "select",
    "create",
    "insert",
    "update",
    "delete",
    "pragma",
    "explain",
    "sqlite3",
    "minhash",
    "lsh",
    "jaccard",
    "blob",
    "similarity",
    "anchor",
    "band",
}
MAX_SNIPPET_LINES = 20


@dataclass(order=True)
class QueryHit:
    rel_path: str
    lineno: int
    repo_name: str
    purpose: str
    mode: str
    snippet: str


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def scan_roots(root: Path, repo_name: str) -> list[Path]:
    include_dirs = INCLUDE_DIRS.get(repo_name, [])
    roots = [root / rel for rel in include_dirs if (root / rel).exists()]
    return roots or [root]


def iter_py_files(root: Path, repo_name: str, progress: bool = False) -> Iterable[Path]:
    seen: set[Path] = set()
    count = 0
    for scan_root in scan_roots(root, repo_name):
        for path in scan_root.rglob("*.py"):
            if path in seen:
                continue
            seen.add(path)
            if should_skip(path):
                continue
            try:
                if path.stat().st_size > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            count += 1
            if progress and count % PROGRESS_EVERY == 0:
                print(f"[scan {repo_name}] {count} files...", file=sys.stderr)
            yield path


def get_call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def contains_keywords(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in SQL_KEYWORDS)


def trim_snippet(text: str) -> str:
    lines = text.strip("\n").splitlines()
    if len(lines) <= MAX_SNIPPET_LINES:
        return "\n".join(lines)
    return "\n".join(lines[:MAX_SNIPPET_LINES] + ["... [trimmed]"])


def node_text(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is not None:
        return segment
    lines = source.splitlines()
    lineno = max(getattr(node, "lineno", 1), 1)
    if lineno <= len(lines):
        return lines[lineno - 1]
    return ""


def nearest_function_name(ancestors: list[ast.AST]) -> str:
    for node in reversed(ancestors):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return "module_level"


def literal_string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("{...}")
        return "".join(parts)
    return None


def extract_hit(
    source: str,
    path: Path,
    repo_root: Path,
    repo_name: str,
    call: ast.Call,
    ancestors: list[ast.AST],
) -> QueryHit | None:
    if get_call_name(call.func) not in CALL_NAMES or not call.args:
        return None

    sql_arg = call.args[0]
    purpose = nearest_function_name(ancestors)
    literal = literal_string_value(sql_arg)
    rel_path = path.relative_to(repo_root).as_posix()

    if literal is not None and contains_keywords(literal):
        return QueryHit(
            rel_path=rel_path,
            lineno=call.lineno,
            repo_name=repo_name,
            purpose=purpose,
            mode="literal",
            snippet=trim_snippet(literal),
        )

    surrounding = node_text(source, call)
    if contains_keywords(surrounding):
        return QueryHit(
            rel_path=rel_path,
            lineno=call.lineno,
            repo_name=repo_name,
            purpose=purpose,
            mode="dynamic",
            snippet=trim_snippet(f"dynamic SQL: {surrounding}"),
        )

    return None


def collect_hits(repo_root: Path, repo_name: str, progress: bool = False) -> list[QueryHit]:
    hits: list[QueryHit] = []
    for path in iter_py_files(repo_root, repo_name, progress=progress):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except Exception:
            continue

        def walk(node: ast.AST, ancestors: list[ast.AST]) -> None:
            if isinstance(node, ast.Call):
                hit = extract_hit(source, path, repo_root, repo_name, node, ancestors)
                if hit is not None:
                    hits.append(hit)
            for child in ast.iter_child_nodes(node):
                walk(child, ancestors + [node])

        walk(tree, [])

    hits.sort(key=lambda hit: (hit.rel_path, hit.lineno))
    return hits


def print_markdown(hits_by_repo: dict[str, list[QueryHit]]) -> None:
    total = sum(len(hits) for hits in hits_by_repo.values())
    print("# SQL Inventory")
    print()
    print("## Summary")
    print()
    print(f"- Total queries found: {total}")
    for repo_name, hits in hits_by_repo.items():
        print(f"- {repo_name}: {len(hits)}")
    print()

    qid = 1
    for repo_name, hits in hits_by_repo.items():
        print(f"## {repo_name}")
        print()
        for hit in hits:
            print(f"### Q{qid}")
            print()
            print(f"- File: `{hit.rel_path}:{hit.lineno}`")
            print(f"- Purpose: `{hit.purpose}`")
            print(f"- Mode: {hit.mode}")
            print("- SQL:")
            print("```sql")
            print(hit.snippet)
            print("```")
            print()
            qid += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print progress every 200 scanned files to stderr.",
    )
    parser.add_argument(
        "--only-repo",
        choices=["nn-gpt", "nn-dataset"],
        help="Scan only one repository.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current_root = Path.cwd().resolve()
    sibling_root = (current_root / ".." / "nn-dataset").resolve()

    repo_roots = {
        current_root.name: current_root,
        sibling_root.name: sibling_root,
    }
    if args.only_repo:
        repo_roots = {args.only_repo: repo_roots[args.only_repo]}

    hits_by_repo = {
        repo_name: collect_hits(repo_root, repo_name, progress=args.progress)
        for repo_name, repo_root in repo_roots.items()
    }
    print_markdown(hits_by_repo)


if __name__ == "__main__":
    main()
