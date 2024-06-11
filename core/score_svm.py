

def score_svm(refdate, windate, gamma=1, C=1):
    from sklearn.svm import SVC
    import numpy as np

    data = np.concatenate((refdate, windate), axis=0)
    y = np.append(
        np.ones(refdate.shape[0]), -np.ones(windate.shape[0])
    )  # 训练SVM的标签, ref为1,win为-1

    # 训练SVM计算统计量
    SVM = SVC(C=C, gamma=gamma).fit(data, y)  # 训练SVM
    M = SVM.decision_function(windate)  # 样本Y到分隔超平面的距离
    score = np.mean(1 / (1 + np.exp(M)))  # 计算统计量
    return score



