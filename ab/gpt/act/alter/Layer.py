import argparse

from ab.gpt.util.NNLayer import alter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=8, help="Maximum number of generation epochs.")
    args = parser.parse_args()
    alter(args.epochs, 'NN_Layers.json', 'deepseek-ai/deepseek-coder-1.3b-instruct')


if __name__ == "__main__":
    main()

