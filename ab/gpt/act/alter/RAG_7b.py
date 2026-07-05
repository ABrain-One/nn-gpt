"""
CLI entry‑point for RAG‑based NN synthesis.
"""

import argparse
from ab.gpt.util.RAG_AlterNN import alter

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
        default="NN_Rag.json", 
        # default="NN_synthesis_rag.json",    
        help="Config JSON filename in conf_test_dir"
    )

    args = parser.parse_args()


    alter(
        epochs=args.epochs,
        test_conf=args.conf,
        llm_name="deepseek-ai/deepseek-coder-7b-instruct" 
    )

if __name__ == "__main__":
    main()
