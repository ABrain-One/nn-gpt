import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(llm_conf='ds_coder_7b_olympic.json', context_length=8192)


if __name__ == '__main__':
    main()
