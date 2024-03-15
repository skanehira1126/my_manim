from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make(
    n_samples: int,
    sensitivity: float,
    pos_weight: float = 0.5,
    var: float = 0.5,
):

    rng = np.random.default_rng()

    # 感応度の計算
    sensitivity_common = np.array([sensitivity] * 2).reshape(1, 2)
    sensitivity_ind = np.array([np.sqrt(1 - sensitivity**2)] * 2).reshape(1, 2)

    # 乱数生成
    common_factor = rng.normal(0, var, (n_samples, 1))
    individual_factor = rng.normal(0, var, (n_samples, 2))

    correlated_random_values = sigmoid(
        common_factor @ sensitivity_common + sensitivity_ind * individual_factor
    )

    prob = correlated_random_values[:, 0]
    label = correlated_random_values[:, 1]

    label = (label >= np.percentile(label, 100 * (1 - pos_weight))).astype(int)

    return label, prob


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    n_sample = 10
    # 良いデータ
    label, prob = make(n_sample, sensitivity=0.8)
    print(roc_auc_score(label, prob))

    # 悪いデータ
    label, prob = make(n_sample, sensitivity=0)
    print(roc_auc_score(label, prob))
