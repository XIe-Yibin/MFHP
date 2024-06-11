def cacul_limit_c(IC_scores, FAP0=0.1, err=0.01):
    """搜索法计算条件控制限"""
    import numpy as np
    from .cacul_CFAP import cacul_CFAP

    init_l = np.percentile(IC_scores.reshape(1, -1), 95)
    CL = init_l
    for k in range(2000):
        CFAP = cacul_CFAP(CL, IC_scores)
        Q_95 = np.percentile(CFAP.reshape(1, -1), 95)

        if Q_95 < FAP0 + err and Q_95 > FAP0 + err:
            break
        elif Q_95 > FAP0:
            CL += 0.1 ** (k // 10)
        elif Q_95 < FAP0:
            CL -= 0.1 ** (k // 10)
    return CL, CFAP


def cacul_limit_u(IC_scores, FAP0=0.1, err=0.01):
    """搜索法计算非条件控制限"""
    import numpy as np
    from .cacul_CFAP import cacul_CFAP

    init_l = np.percentile(IC_scores.reshape(1, -1), 50)
    UL = init_l
    for k in range(1000):
        CFAP = cacul_CFAP(UL, IC_scores)
        Q_50 = np.percentile(CFAP.reshape(1, -1), 50)

        if Q_50 < FAP0 + err and Q_50 > FAP0 + err:
            break
        elif Q_50 > FAP0:
            UL += 0.1 ** (k // 10)
        elif Q_50 < FAP0:
            UL -= 0.1 ** (k // 10)
    return UL, CFAP
