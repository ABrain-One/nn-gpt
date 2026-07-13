import argparse

from ab.gpt.util.AlterNN import alter


# Purpose: Verify that batch generation works correctly.
# Issue: [EXTRACT] ✗ No NN code found
# [INFO]Response Invalid!

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=8, help="Maximum number of generation epochs.")
    args = parser.parse_args()
    alter(args.epochs, 'NN_alter.json', 'open-r1/OlympicCoder-7B', batch_size=8)


if __name__ == "__main__":
    main()
