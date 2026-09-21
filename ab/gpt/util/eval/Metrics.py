import evaluate
import numpy as np


class Metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_bleu(self, eval_preds):
        """
        eval_preds:
          - predictions: logits or token IDs
          - labels: token IDs with -100 as ignore index
        Then:
          1) convert logits -> token IDs (argmax if needed)
          2) replace -100 -> pad_id
          3) decode to text
          4) pass plain strings to BLEU metric
        """
        bleu_metric = evaluate.load("bleu")

        predictions, labels = eval_preds

        # ---- 1) Handle tuples from Trainer ----
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if isinstance(labels, tuple):
            labels = labels[0]

        predictions = np.asarray(predictions)
        labels = np.asarray(labels)

        # ---- 2) If predictions are logits, take argmax over vocab dimension ----
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        # ---- 3) Replace ignore index (-100) in labels with pad token id ----
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        labels = np.where(labels != -100, labels, pad_id)

        # ---- 4) Decode to plain strings (NO manual token splitting) ----
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )

        # BLEU / sacrebleu in `evaluate` expects:
        #   predictions: List[str]
        #   references:  List[List[str]] (each inner list = list of references)
        references = [[lbl] for lbl in decoded_labels]

        result = bleu_metric.compute(
            predictions=decoded_preds,
            references=references,
        )

        # Some versions return "bleu", some return "score"
        bleu_value = result.get("bleu", result.get("score", 0.0))
        return {"bleu": bleu_value}

    def init_compute_metrics(self, test_metric: str | None):
        metric_switch = {
            "bleu": self.compute_bleu,
        }
        if test_metric:
            if test_metric not in metric_switch:
                raise ValueError(f"Unsupported metric: {test_metric}")
            return metric_switch[test_metric]
        else:
            return None
