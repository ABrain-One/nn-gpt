import argparse

from ab.gpt.util.nn.AlterNN import alter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=8, help="Maximum number of generation epochs.")
    parser.add_argument('--vllm_url', type=str, default=None,
                        help="Use a remote OpenAI-compatible vLLM server instead of the local LLM, "
                             "e.g. http://132.187.14.67:30004/v1")
    parser.add_argument('--vllm_model', type=str, default=None,
                        help="Served model id (default: first model listed by the server).")
    parser.add_argument('--vllm_workers', type=int, default=4, help="Parallel requests to the vLLM server.")
    args = parser.parse_args()
    alter(args.epochs, 'NN_alter.json', 'open-r1/OlympicCoder-7B',
          vllm_url=args.vllm_url, vllm_model=args.vllm_model, vllm_workers=args.vllm_workers)


if __name__ == "__main__":
    main()
