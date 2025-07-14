import argparse
from .util.rag_AlterNN import alter

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-e', '--epochs', type=int, default=8,
                   help="Number of generation epochs")
    p.add_argument('-c', '--conf', type=str, default='NN_synthesis_rag.json',
                   help="Config JSON filename in conf_test_dir")
    args = p.parse_args()

    alter(args.epochs, 'NN_synthesis_rag.json', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')

if __name__ == "__main__":
    main()
