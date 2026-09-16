# src/eval/compute_tier2.py
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from eval.correctness import is_correct

LOG_PATH = "data/processed/eval_results.jsonl"
SYSTEMS_WITH_UNCERTAINTY = {
    "adaptive": lambda result: result.get("_cached_features", {}).get("uncertainty"),
    "fixed_threshold": lambda result: result.get("uncertainty"),
}


def expected_calibration_error(confidences, correctness, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = [(c, y) for c, y in zip(confidences, correctness) if lo <= c < hi or (i == n_bins - 1 and c == hi)]
        if not in_bin:
            continue
        bin_conf = np.mean([c for c, _ in in_bin])
        bin_acc = np.mean([y for _, y in in_bin])
        ece += (len(in_bin) / n) * abs(bin_acc - bin_conf)
    return ece


def compute_tier2():
    rows = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    for system, extractor in SYSTEMS_WITH_UNCERTAINTY.items():
        confidences, correctness = [], []
        for row in rows:
            if row["system"] != system:
                continue
            result = row["result"]
            if "error" in result:
                continue
            uncertainty = extractor(result)
            if uncertainty is None:
                continue
            predicted = result.get("final_answer")
            correct = is_correct(predicted, row.get("acceptable_answers", []), row.get("ground_truth_answer"))
            if correct is None:
                continue
            confidences.append(1 - uncertainty)  # confidence = 1 - uncertainty
            correctness.append(1 if correct else 0)

        if len(set(correctness)) < 2:
            print(f"{system}: not enough class diversity for AUROC (need both correct and incorrect examples in sample)")
            continue

        ece = expected_calibration_error(confidences, correctness)
        brier = brier_score_loss(correctness, confidences)
        auroc = roc_auc_score(correctness, confidences)

        print(f"\n{system}:")
        print(f"  ECE:    {ece:.3f}")
        print(f"  Brier:  {brier:.3f}")
        print(f"  AUROC:  {auroc:.3f}  (confidence separating correct vs incorrect)")
        print(f"  n = {len(confidences)}")


if __name__ == "__main__":
    compute_tier2()