# -*- coding: utf-8 -*-


def woe_replace(df, col, bin_dict):
    '''
    将df的值用woe替换；

    :param df: df，数据集（应和分箱、计算woe值用的df一致）；
    :param col: str，将要替换woe值的列的列名（须为数值型）；
    :param bin_dict: dict，key是各箱索引，vlaue是各箱对应的下边界，上边界，woe，iv构成的列表；{0:['-inf', 50.0, 1.19, 0.006]}
    :return: df，原df（值已用woe替换)；
    '''

    length = len(bin_dict)  # 字典的长度（首箱，[其他箱*]，尾箱，null箱）

    for key in bin_dict.keys():     #下边界[0]，上边界[1]，woe[2]，iv[3]
        if key == 0:
            df[col][df[col] <= bin_dict[0][1]] = bin_dict[0][2]   #首箱
        elif key == length - 2:
            df[col][bin_dict[length - 2][0] <= df[col]] = bin_dict[length - 2][2]     #尾箱
        elif key == length - 1:
            df[col][df[col].isnull()] = bin_dict[length - 1][2]     #null箱
        else:
            df[col][(bin_dict[key][0] <= df[col]) & (df[col] <= bin_dict[key][1])] = bin_dict[key][2]   #其他箱*

    return df
