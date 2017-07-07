"""
This implements an offline evaluation framework of E-E policies,
which mimics a scaled down version of the production system.
Ref: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/EE_Autosuggest_MSRTR.pdf
"""
import numpy as np


def check_sess_click(y_true, y_predict, k):
    """
    :param y_true: np.1darray, label for click (1 for clicked, 0 otherwise)
    :param y_predict: list, model's predicted CTR for each item.
    :param k: top-k item as a scaled down session. k <= 10.
    :return: num of item clicked among top-k.
    """
    assert k <= len(y_true)
    arg = np.argsort(y_predict)[::-1][:k]  # get index of top-k in y_predict
    top_k_label = y_true[arg]
    return sum(top_k_label)


def log_k_items(x, y, exp_idx, k):

    assert k < x.shape[0] and exp_idx < x.shape[0]
    x_topk = x[:k]
    y_topk = y[:k]

    x_topk[k-1] = x[exp_idx]
    y_topk[k-1] = y[exp_idx]
    return x_topk, y_topk


if __name__ == '__main__':
    x = np.random.randint(low=0, high=10, size=(10, 5))
    y = np.random.randint(low=0, high=2, size=10)
    print(x, y)
    x_exp, y_exp = log_k_items(x, y, 8, 5)
    assert x_exp.shape[0] == 5
    print(x_exp, y_exp)
