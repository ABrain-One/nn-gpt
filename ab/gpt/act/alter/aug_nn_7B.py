import argparse
import gc
import subprocess
import sys

from ab.gpt.util.nn.AlterAugNN import alter_aug


def main():
    parser = argparse.ArgumentParser()
    # generation
    parser.add_argument('-e', '--epochs', type=int, default=10, help="Generation epochs.")
    parser.add_argument('-p', '--prompts', type=int, default=10, help="Schedule+network prompts per epoch.")
    parser.add_argument('--pool_top', type=int, default=50,
                        help="Reference networks are drawn from the best N LEMUR networks.")
    parser.add_argument('--nn_prefixes', nargs='*', default=None,
                        help="Only use LEMUR networks whose name starts with one of these (e.g. ResNet).")
    parser.add_argument('--max_ref_chars', type=int, default=6000,
                        help="Skip reference networks whose code is longer than this (two go in one prompt).")
    parser.add_argument('--dataset', type=str, default='imagenette')
    parser.add_argument('--llm', type=str, default='open-r1/OlympicCoder-7B')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--vllm_url', type=str, default='http://132.187.14.67:30004/v1',
                        help="http://132.187.14.67:30004/v1")
    parser.add_argument('--vllm_model', type=str, default=None)
    parser.add_argument('--vllm_workers', type=int, default=4)
    args = parser.parse_args()

    alter_aug(args.epochs, 'Augment_NN_gen.json', args.llm, n_prompts=args.prompts, pool_top=args.pool_top,
              nn_prefixes=args.nn_prefixes, dataset=args.dataset, max_ref_chars=args.max_ref_chars,
              seed=args.seed, vllm_url=args.vllm_url, vllm_model=args.vllm_model,
              vllm_workers=args.vllm_workers)


if __name__ == "__main__":
    main()