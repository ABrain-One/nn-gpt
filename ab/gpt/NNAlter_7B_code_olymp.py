import argparse

from ab.gpt.util.AlterNN import alter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=8, help="Maximum number of generation epochs.")
    parser.add_argument(
        '--analogical',
        action='store_true',
        help="Prepend a tracked analogical exemplar to the standard NNAlter prompt.",
    )
    args = parser.parse_args()
    alter(args.epochs, 'NN_alter.json', 'open-r1/OlympicCoder-7B', analogical=args.analogical)


if __name__ == "__main__":
    main()
