## SVM统计量
def svm_score_matrix(RefDate, MonDate, ref_size, win_size, I, K, gamma=1, C=1):
    """
    计算一个过程的统计量
    输入：
    RefDat: 参考样本， 最后一列是标签1
    WinDate: 监控样本， 最后一列是标签-1
    RefDate.shape[0]>= ref_size*I, MonDate.shape[0]>=I*win_size
    输出：得分score
    """
    import numpy as np
    from .score_svm import score_svm
    from .score_svm_weight import score_svm_weight

    ## K为仿真次数, I为检测次数
    score_matrix = np.zeros((K, I))
    ## 利用SVM求解统计量
    for k in range(K):
        RefDate_ = RefDate[ref_size * k : ref_size * (k + 1)]
        for j in range(I):
            MonDate_ = MonDate[win_size * j : win_size * (j + 1)]
            score_matrix[k, j] = score_svm(RefDate_, MonDate_, gamma, C)
    return score_matrix


## ranksum统计量
def ranksum_score_matrix(RefDate, MonDate, ref_size, win_size, I, K):
    """
    计算一个过程的统计量
    输入：
    RefDat: 参考样本， 最后一列是标签1
    WinDate: 监控样本， 最后一列是标签-1
    RefDate.shape[0]>= ref_size*I, MonDate.shape[0]>=I*win_size
    输出：得分score
    """
    import numpy as np
    from .score_ranksum import score_ranksum

    ## K为仿真次数, I为检测次数
    score_matrix = np.zeros((K, I))
    ## 利用SVM求解统计量
    for k in range(K):
        RefDate_ = RefDate[ref_size * k : ref_size * (k + 1)]
        for j in range(I):
            MonDate_ = MonDate[win_size * j : win_size * (j + 1)]
            score_matrix[k, j] = score_ranksum(RefDate_, MonDate_)
    return score_matrix

## ranksum统计量
def ranksum_2_score_matrix(RefDate, MonDate, ref_size, win_size, I, K):
    import numpy as np
    from .score_ranksum import score_ranksum_2

    ## K为仿真次数, I为检测次数
    score_matrix = np.zeros((K, I))
    ## 利用SVM求解统计量
    for k in range(K):
        RefDate_ = RefDate[ref_size * k : ref_size * (k + 1)]
        for j in range(I):
            MonDate_ = MonDate[win_size * j : win_size * (j + 1)]
            score_matrix[k, j] = score_ranksum_2(RefDate_, MonDate_)
    return score_matrix


## SVM-weight统计量
def svm_weight_score_matrix(RefDate, MonDate, ref_size, win_size, I, K, gamma=1, C=1):
    """
    计算一个过程的统计量
    输入：
    RefDat: 参考样本， 最后一列是标签1
    WinDate: 监控样本， 最后一列是标签-1
    RefDate.shape[0]>= ref_size*I, MonDate.shape[0]>=I*win_size
    输出：得分score
    """
    import numpy as np
    from .score_svm_weight import score_svm_weight

    ## K为仿真次数, I为检测次数
    score_matrix = np.zeros((K, I))
    ## 利用SVM求解统计量
    for k in range(K):
        RefDate_ = RefDate[ref_size * k : ref_size * (k + 1)]
        for j in range(I):
            MonDate_ = MonDate[win_size * j : win_size * (j + 1)]
            score_matrix[k, j] = score_svm_weight(RefDate_, MonDate_, gamma, C)
    return score_matrix
