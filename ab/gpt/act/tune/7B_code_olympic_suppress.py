import ab.gpt.act.tune.Tune as TuneNNGen


def main():
    TuneNNGen.main(
        llm_conf='ds_coder_7b_olympic_4096.json',
        suppress_thinking=True,
    )


if __name__ == '__main__':
    main()
