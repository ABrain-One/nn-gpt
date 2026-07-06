import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(llm_conf='ds_qwen3_8b.json', context_length=4096)


if __name__ == '__main__':
    main()
