import numpy as np
from sklearn.metrics import average_precision_score


def macro_pr_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    categories: np.ndarray | list[str],
) -> float:
    """
    Competition metric: mean of per-category Average Precision (PR-AUC).
    :param y_true: binary targets (0/1)
    :param scores: raw prediction scores (any order-preserving scale)
    :param categories: category per pair (same length as y_true)
    :return: macro PR-AUC
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    categories = np.asarray(categories)
    per_category = []
    for cat in np.unique(categories):
        mask = categories == cat
        per_category.append(average_precision_score(y_true[mask], scores[mask]))
    return float(np.mean(per_category))
