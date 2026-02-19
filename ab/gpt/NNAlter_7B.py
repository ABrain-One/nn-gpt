import argparse

from ab.gpt.util.AlterNN import alter
from ab.gpt.util.Const import nngpt_upload, base_llm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=8, help="Maximum number of generation epochs.")
    parser.add_argument('-n', '--num-supporting-models', type=int, default=1, help="Number of supporting models to fetch from database for more ideas.")
    args = parser.parse_args()
    stacked_model = nngpt_upload / base_llm
    alter(args.epochs, 'NN_alter.json', str(stacked_model), n=args.num_supporting_models, temperature=0.8, top_k=100)


if __name__ == "__main__":
    main()
