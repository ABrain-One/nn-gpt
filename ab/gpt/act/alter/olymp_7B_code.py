import argparse

from ab.gpt.util.AlterNN import alter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=8, help="Maximum number of generation epochs.")
    args = parser.parse_args()
    alter(args.epochs, 'NN_alter.json', 'open-r1/OlympicCoder-7B')


if __name__ == "__main__":
    main()
