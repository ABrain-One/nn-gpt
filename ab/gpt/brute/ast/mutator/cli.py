#!/usr/bin/env python
"""Command-line entrypoint for the brute-force architecture-mutation search.

Runs the mutator WITHOUT editing config.py: flags are forwarded to the engine as
env vars (config.py reads them), which also propagate to multiprocessing workers.

Examples:
  # isolate the skip operator on one seed (the weak-seed residual experiment)
  python cli.py --lemur-root /path/to/nn-dataset --models DenoiseUnet2F --only skip --attempts 2 --debug

  # a mixed search over several seeds
  python cli.py --lemur-root /path/to/nn-dataset \
      --models DenoiseUnet7F,DenoiseUnet9F --weights skip:1.0,activation:0.5 --attempts 50

  # just show what operators exist
  python cli.py --list-operators

Notes:
  * --lemur-root must point at the nn-dataset repo (the LEMUR DB is read from there,
    and seeds must exist in it). Falls back to $NN_DATASET_ROOT, else current dir.
  * Generated model .py files land under config.MUTATED_MODELS_OUTPUT_ROOT
    (override with --output). Evaluate/rank them with denoise_search.py.
"""
import argparse
import os
import sys

# Make the `mutator` package importable regardless of the current directory.
_AST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AST_DIR not in sys.path:
    sys.path.insert(0, _AST_DIR)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Brute-force neural-architecture mutation search (LEMUR seeds).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--models", help="comma-separated seed model names (LEMUR nn names)")
    p.add_argument("--attempts", type=int, help="mutation attempts per model")
    p.add_argument("--only", help="isolate a single operator (weight 1.0, others 0.0)")
    p.add_argument("--weights", help="per-operator weights, e.g. 'skip:1.0,activation:0.5'")
    p.add_argument("--output", help="output root for generated model .py files")
    p.add_argument("--workers", type=int, help="max parallel worker processes")
    p.add_argument("--debug", action="store_true", help="single-process + verbose (recommended for first runs)")
    p.add_argument("--lemur-root", help="path to the nn-dataset repo (LEMUR root)")
    p.add_argument("--list-operators", action="store_true", help="list mutation operators and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    # LEMUR root must be both importable (for `import ab.nn`) and the cwd (the LEMUR
    # DB root is resolved from cwd) BEFORE the orchestrator imports ab.nn.
    root = args.lemur_root or os.environ.get("NN_DATASET_ROOT")
    if root:
        root = os.path.abspath(root)
        os.chdir(root)
        if root not in sys.path:
            sys.path.insert(0, root)  # so `import ab.nn` resolves without PYTHONPATH

    # Map flags -> env vars consumed by config.py (these survive multiprocessing).
    if args.models:
        os.environ["BRUTE_MODELS"] = args.models
    if args.attempts is not None:
        os.environ["BRUTE_ATTEMPTS"] = str(args.attempts)
    if args.only:
        os.environ["BRUTE_ONLY"] = args.only
    if args.weights:
        os.environ["BRUTE_WEIGHTS"] = args.weights
    if args.output:
        os.environ["BRUTE_OUTPUT"] = args.output
    if args.workers is not None:
        os.environ["BRUTE_WORKERS"] = str(args.workers)
    if args.debug:
        os.environ["BRUTE_DEBUG"] = "1"

    from mutator import config

    if args.list_operators:
        print("Mutation operators (name  weight  description):")
        for name in config.MUTATION_TYPE_WEIGHTS:
            weight = config.MUTATION_TYPE_WEIGHTS[name]
            desc = config.OPERATOR_DESCRIPTIONS.get(name, "")
            print(f"  {name:14s} {weight:<5} {desc}")
        return

    from mutator.orchestrator import main as orchestrator_main
    orchestrator_main()


if __name__ == "__main__":
    main()
