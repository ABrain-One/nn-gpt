import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(llm_conf='nngpt_dsr1_distill_qwen_7b_r.json', context_length=4096)


if __name__ == '__main__':
    main()
