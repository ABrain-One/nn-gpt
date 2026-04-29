import argparse

from ab.gpt.brute.fract.backbone.NNAlterBN import alter_diverse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Maximum number of generation epochs.')
    parser.add_argument(
        '--variants-per-pattern',
        type=int,
        default=20,
        help='Number of generated models per diverse backbone pattern.',
    )
    args = parser.parse_args()
    alter_diverse(args.epochs, variants_per_pattern=args.variants_per_pattern)


if __name__ == '__main__':
    main()
