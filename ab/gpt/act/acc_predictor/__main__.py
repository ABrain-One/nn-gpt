"""End-to-end pipeline entry point: python -m ab.gpt.act.acc_predictor."""
from __future__ import annotations

from .config import RAW_INPUT_PATH, RAW_CSV_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_TEST_OUTPUT_PATH
from .preprocess import data_preprocessing
from .dataset import prepare_llm_datasets
from .train import train_model
from .eval import test_model




def main() -> None:
    print("=" * 80)
    print("Step 1/4: Data preprocessing (nn_dataset → JSONL)")
    print("=" * 80)
    data_preprocessing(
        output_jsonl_path=RAW_INPUT_PATH,
        output_csv_path=RAW_CSV_PATH,
    )

    print("\n" + "=" * 80)
    print("Step 2/4: Prepare LLM training datasets")
    print("=" * 80)
    train_path, val_path, test_path = prepare_llm_datasets()
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Test:  {test_path}")

    print("\n" + "=" * 80)
    print("Step 3/4: Fine-tune model")
    print("=" * 80)
    train_model(train_path=train_path, val_path=val_path)
    print(f"Model saved to {DEFAULT_OUTPUT_DIR}")

    print("\n" + "=" * 80)
    print("Step 4/4: Test model on held-out test set")
    print("=" * 80)
    test_model(
        model_path=DEFAULT_OUTPUT_DIR,
        data_path=test_path,
        output_path=DEFAULT_TEST_OUTPUT_PATH,
    )
    print(f"\nDone. Pipeline complete.")

if __name__ == "__main__":
    main()
