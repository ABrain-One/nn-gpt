import argparse

from ab.gpt.util.AlterNN import alter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=8, help="Maximum number of generation epochs.")
    parser.add_argument('-n', '--num-supporting-models', type=int, default=1, help="Number of supporting models to fetch from database for more ideas.")
    args = parser.parse_args()
    alter(args.epochs, 'NN_alter.json', 'openai/gpt-oss-20b', n=args.num_supporting_models, temperature=0.8, top_k=100, inference_gpt_oss=True, inference_gpt_oss_max_input_length=4096)


if __name__ == "__main__":
    main()
