import numpy as np
from .score_svm import score_svm
from .score_svm_weight import score_svm_weight
from .score_matrix import (
    svm_score_matrix,
    ranksum_score_matrix,
    ranksum_2_score_matrix,
    svm_weight_score_matrix,
)
from .cacul_CFAP import cacul_CFAP
from .cacul_CRSP import cacul_CRSP
from .cacul_limit import cacul_limit_c, cacul_limit_u
from .chart_preform import chart_preform
from .score_ranksum import score_ranksum,score_ranksum_2
from .kernel_func import linear, poly, rbf
from .svm_smo_classifier import SVMClassifier


def read_limit(filename):
    file = open(filename, "r")  # 打开文件
    file_data = file.readlines()  # 读取所有行

    UL_list = []
    CL_list = []
    for i, row in enumerate(file_data[1:]):
        tmp_list = row.split(" ")[4]  # 按‘，’切分每行的数据
        if i % 2 == 0:  # 第一行是conditional
            CL_list += [tmp_list]
        else:
            UL_list += [tmp_list]
    CL_list = np.array(CL_list, dtype=np.float64)
    UL_list = np.array(UL_list, dtype=np.float64)
    return CL_list, UL_list
