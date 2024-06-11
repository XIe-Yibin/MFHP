def cacul_CRSP(L, OC_scores):
    """计算正确报警概率CRSP
    输出：长度为R的向量"""

    import numpy as np

    L = L
    R, K, I = OC_scores.shape[0], OC_scores.shape[1], OC_scores.shape[2]

    P0 = np.zeros(R)
    for r in range(R):
        # 误报率
        P0[r] = np.sum(OC_scores[r, :, :] > L) / (I * K)

    kappa = 1
    CRSP = 1 - (1 - P0) ** (I - kappa + 1)

    return CRSP
