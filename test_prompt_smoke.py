import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from ab.gpt.util.prompt.NNGenCurriculumPrompt import NNGenPrompt  # adjust import path if yours differs


def main():
    prompts_path = Path("ab/gpt/conf/prompt/train/NN_gen_curriculum.json")
    assert prompts_path.exists(), f"Missing {prompts_path}"

    tok = AutoTokenizer.from_pretrained("ABrain/NNGPT-UniqueArch-Rag", use_fast=True)  # any available tokenizer
    p = NNGenPrompt(max_len=2048, tokenizer=tok, prompts_path=str(prompts_path))

    # very small cap so it runs fast
    df = p.get_raw_dataset(only_best_accuracy=True, n_training_prompts=5)

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, "No prompts generated"
    for c in ["instruction", "response", "text"]:
        assert c in df.columns

    # sanity: text actually contains the XML tags we expect
    sample = df.iloc[0]["text"]
    assert "<nn>" in sample and "</nn>" in sample, "Prompt output missing <nn> tag"

    print("[PASS] prompt smoke: rows=", len(df))
    print(df[["instruction", "response"]].head(1).to_string(index=False))


if __name__ == "__main__":
    main()
