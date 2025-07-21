# NNAlterRag.py
"""
CLI entry‑point for RAG‑based NN synthesis.
"""

import argparse
from .util.rag_AlterNN import alter      # ← correct import path

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate altered neural‑network architectures with RAG."
    )
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=8,
        help="Number of generation epochs"
    )
    parser.add_argument(
        "-c", "--conf",
        type=str,
        default="NN_Rag.json",      # ← your new JSON file
        help="Config JSON filename in conf_test_dir"
    )
    args = parser.parse_args()

    alter(
        epochs=args.epochs,
        test_conf=args.conf,
        llm_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )

if __name__ == "__main__":
    main()
