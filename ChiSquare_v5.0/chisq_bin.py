# -*- coding: utf-8 -*-
import numpy as np



def chisq_bin_continuous(df, feature, target, n_bins=10, min_samples=None, chi_threshold=None, balance=True):
    '''
    连续型特征的卡方分箱(须提前对类别进行排序)

    :param df: dataframe；数据集
    :param feature: str; 待分箱的特征；
    :param target: str; y标签；
    :param n_bins: int; 最大分箱数；
    :param min_samples: int or float; 每箱最小样本数（当为float时，须小于1，视为百分比）；
    :param chi_threshold: float; 卡方值阈值；
    :param balance: bool; 是否平衡处理；
    :return: list; 切分点列表；
    '''
    feature_data = df[feature]
    target_data = df[target]

    if min_samples and min_samples < 1:
        min_samples = len(feature_data) * min_samples

    feature_data = feature_data.fillna(-1)

    # 计算各类别的value_counts，并形成映射
    feature_unique = np.sort(feature_data.unique())
    len_f = len(feature_unique)
    grouped_agg = {}
    grouped_map = {}
    for r in range(len_f):
        tmp = target_data[feature_data == feature_unique[r]]
        grouped_agg[r] = [(tmp == 0).sum(), (tmp == 1).sum()]
        grouped_map[r] = [feature_unique[r]]

    # 处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
    for i in range(len(grouped_agg) - 2):
        if (grouped_agg[i][0] + grouped_agg[i + 1][0] == 0) or (grouped_agg[i][1] + grouped_agg[i + 1][1] == 0):
            grouped_agg[i + 1][0] += grouped_agg[i][0]
            grouped_agg[i + 1][1] += grouped_agg[i][1]
            grouped_map[i + 1] += grouped_map[i]
            grouped_map.pop(i)
            grouped_agg.pop(i)

    while 1:
        # 分箱数 < n_bins时
        if n_bins and len(grouped_agg) <= n_bins:
            break

        # 各分箱的最小样本数 > min_samples时
        grouped_samples = [v[0]+v[1] for v in grouped_agg.values()]
        if min_samples and min(grouped_samples) > min_samples:
            break

        # 计算相邻区间卡方值
        min_chi = np.inf
        keys = list(grouped_agg.keys())
        for i in range(len(grouped_agg) - 2):
            cols = [grouped_agg[keys[i]][0] + grouped_agg[keys[i + 1]][0], grouped_agg[keys[i]][1] + grouped_agg[keys[i + 1]][1]]
            rows = [sum(grouped_agg[keys[i]]), sum(grouped_agg[keys[i + 1]])]
            total = sum(rows)

            chi = 0
            for j in range(2):
                for k in range(2):
                    e = rows[j] * cols[k] / total
                    chi += (grouped_agg[keys[i + j]][0 + k] - e) ** 2 / e

            if balance:
                chi *= total

            if chi <= min_chi:
                min_chi = chi
                min_ix = i

        if chi_threshold and min_chi > chi_threshold:
            break

        min_key = keys[min_ix]
        drop_key = keys[min_ix + 1]
        grouped_agg[min_key][0] += grouped_agg[drop_key][0]
        grouped_agg[min_key][1] += grouped_agg[drop_key][1]
        grouped_map[min_key] += grouped_map[drop_key]
        grouped_agg.pop(drop_key)
        grouped_map.pop(drop_key)

    # print(grouped_map)
    cut_nodes = [min(l) for l in grouped_map.values()][1:]

    return cut_nodes


