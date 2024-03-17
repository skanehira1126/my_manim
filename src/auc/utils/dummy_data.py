from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make(
    n_samples: int,
    sensitivity_list: list[float],
    pos_weight: float = 0.5,
    var: float = 0.5,
):

    rng = np.random.default_rng()

    # 乱数生成
    common_factor = rng.normal(0, var, (n_samples, 1))
    label = (common_factor >= np.percentile(common_factor, 100 * (1 - pos_weight))).astype(int)

    rand_values = []
    for sensitivity in sensitivity_list:
        # 感応度の計算
        sensitivity_common = np.array([[sensitivity]])
        sensitivity_ind = np.array([[np.sqrt(1 - sensitivity**2)]])

        individual_factor = rng.normal(0, var, (n_samples, 1))
        correlated_random_values = sigmoid(
            common_factor @ sensitivity_common + sensitivity_ind * individual_factor
        )

        rand_values.append(correlated_random_values.ravel())

    return label.ravel(), rand_values


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    n_sample = 10
    label, probs = make(n_sample, sensitivity_list=[0, 0.8])
    print(roc_auc_score(label, probs[0]))
    print(roc_auc_score(label, probs[1]))
