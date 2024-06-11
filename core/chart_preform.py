def chart_preform(FAP):
    """计算SDFAP,CFAP_95,CFAP_75,CFAP_50"""
    import numpy as np

    FAP_50 = np.percentile(FAP.reshape(1, -1), 50)
    FAP_75 = np.percentile(FAP.reshape(1, -1), 75)
    FAP_95 = np.percentile(FAP.reshape(1, -1), 95)
    SDFAP = np.var(FAP.reshape(1, -1))
    return FAP_50, FAP_75, FAP_95, SDFAP
