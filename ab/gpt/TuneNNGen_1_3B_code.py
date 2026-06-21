import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(llm_conf='ds_coder_1.3b_instruct.json', max_new_tokens=12 * 1024, enable_merge=False)


if __name__ == '__main__':
    main()
