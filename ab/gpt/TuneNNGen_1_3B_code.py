import ab.gpt.TuneNNGen as TuneNNGen


def main():
    TuneNNGen.main(llm_conf='nngpt_ds_coder_1.3b_instruct.json', max_new_tokens=4096,enable_merge=True)


if __name__ == '__main__':
    main()
