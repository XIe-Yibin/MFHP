import numpy as np

## MWR_I
def score_ranksum(refdate, windate):
    # 计算ranksum统计量
    data = np.concatenate((refdate, windate), axis=0)
    rank = np.argsort(data, axis=0)
    ranksum = (
        np.sum(rank[-windate.shape[0] :, :], axis=0)
        - windate.shape[0] * (windate.shape[0] + refdate.shape[0] + 1) / 2
    )
    z = ranksum / np.sqrt(
        windate.shape[0]
        * refdate.shape[0]
        * (windate.shape[0] + refdate.shape[0] + 1)
        / 12
    )
    score = np.sum(z**2)
    return score 

## MWR_II
def score_ranksum_2(refdate, windate):

    bar_s0 = np.mean(refdate, axis=0)
    cov_s0 = np.cov(refdate, rowvar=False)
    inv_cov_s0 = np.linalg.inv(cov_s0)


    d_i_j = np.array([
        (s_i_j - bar_s0).T @ inv_cov_s0 @ (s_i_j - bar_s0)
        for s_i_j in windate
    ])

    o_r = np.array([
        (s_0_r - bar_s0).T @ inv_cov_s0 @ (s_0_r - bar_s0)
        for s_0_r in refdate
    ])

    # Compute indicator matrix
    indicator_matrix = np.array([
        [1 if d_ij > o_rk else 0 for o_rk in o_r]
        for d_ij in d_i_j
    ])

    # Calculate the MWR-II score
    U = np.sum(indicator_matrix) - len(refdate) * len(windate) / 2
    score = U / np.sqrt(len(refdate) * len(windate) * (len(refdate) + len(windate) + 1) / 12)
    return score


if __name__ == '__main__':
    refdate = np.random.rand(10, 3) 
    windate = np.random.rand(8, 3)   
    print(score_ranksum(refdate, windate))
    print(score_ranksum_2(refdate, windate))  





