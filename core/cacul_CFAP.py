def cacul_CFAP(L, IC_scores):
    """计算误报概率CFAP
    输出：长度为R的向量"""

    import numpy as np

    L = L
    R, K, I = IC_scores.shape[0], IC_scores.shape[1], IC_scores.shape[2]
    P0 = np.zeros(R)
    for r in range(R):
        # 误报率
        P0[r] = np.sum(IC_scores[r, :, :] > L) / (I * K)
    CFAP = 1 - (1 - P0) ** I
    return CFAP
