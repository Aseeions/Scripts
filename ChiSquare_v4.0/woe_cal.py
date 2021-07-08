import math
import pandas as pd

def woe_cal(df, col, target, cutNodes):
    '''
    计算各bin的woe值，iv值；

    :param df: df, 包含标签列的数据集；
    :param col: str, 要计算woe的列的列名（须为数值型）；
    :param target: str, 标签列的列名；
    :param cutNodes: list, 分箱的切割点的列表（不含首尾）；

    :return: dict，key是各箱，vlaue是各箱对应的下边界，上边界，woe，iv；
    '''

    bin_data = {}       #{0:df[x]}；每箱对应的分df；
    # bin_len = {}        #{0:12}
    bin_lower = {}      #{0:'-inf'}；每箱的下边界
    bin_upper = {}      #{0:'inf'}；每箱的上边界
    bin_count_0 = {}    #{0:10}
    bin_count_1 = {}    #{0:2}
    bin_woe = {}        #{0:3.1415}；每箱的woe；
    bin_iv = {}         #{0:0.005}；每箱的iv；
    total_dict = {}     #{'bin_data':'bin_lower':bin_lower, 'bin_upper':bin_upper, 'bin_woe':bin_woe, 'bin_iv':bin_iv}
    sum_iv = 0.0

    if len(cutNodes) == 1:    #只有一个cutNode时
        bin_data[0] = df.loc[df[col] <= float(cutNodes[0]), :]  #左边界箱数据
        bin_data[1] = df.loc[df[col] > float(cutNodes[0]), :]   #右
        bin_data[2] = df.loc[df[col].isnull()]      #空值箱

        # bin_len[0] = len(bin_data[0])   #每箱的数据个数
        # bin_len[1] = len(bin_data[1])
        # bin_len[2] = len(bin_data[2])

        bin_lower[0] = '-inf'           #每箱的下界
        bin_lower[1] = cutNodes[0]
        bin_lower[2] = 'null'

        bin_upper[0] = cutNodes[0]      #每箱的上界
        bin_upper[1] = 'inf'
        bin_upper[2] = 'null'

    else:    #cutNode个数大于1时
        for index, cutNode in enumerate(cutNodes):      #cutNode = cutNodes[index]
            if index == 0:      #左边界箱
                bin_data[0] = df.loc[df[col] <= float(cutNode), :]
                # bin_len[0] = len(bin_data[0])
                bin_lower[0] = '-inf'
                bin_upper[0] = float(cutNode)

            else:       #中间箱
                bin_data[index] = df.loc[(df[col] > float(cutNodes[index - 1])) & (df[col] <= float(cutNode)),:]
                # bin_len[index] = len(bin_data[index])
                bin_lower[index] = cutNodes[index - 1]
                bin_upper[index] = float(cutNode)

        bin_data[index + 1] = df.loc[df[col] > float(cutNode), :]   #右边界箱
        # bin_len[index + 1] = len(bin_data[index + 1])
        bin_lower[index + 1] = cutNode
        bin_upper[index + 1] = 'inf'

        bin_data[index + 2] = df.loc[df[col].isnull()]      #空值箱
        # bin_len[index + 2] = len(bin_data[index + 2])
        bin_lower[index + 2] = 'null'
        bin_upper[index + 2] = 'null'

    count_0 = len(df[df[target] == 0])      #df中总负样本数量
    count_1 = len(df[df[target] == 1])      #正

    for index, item in enumerate(bin_data):
        bin_count_0[index] = len(bin_data[index][bin_data[index][target] == 0])     #各箱中负样本数量
        bin_count_1[index] = len(bin_data[index][bin_data[index][target] == 1])     #正

        ratio_bin_0 = float(bin_count_0[index]) / float(count_0)    #每箱/总（负样本）
        ratio_bin_1 = float(bin_count_1[index]) / float(count_1)    #正

        try:
            bin_woe[index] = math.log((ratio_bin_1 + 0.0001) / (ratio_bin_0 + 0.0001))      #计算各箱woe
            bin_iv[index] = bin_woe[index] * (ratio_bin_1 - ratio_bin_0)  #计算各箱iv

        except Exception as e:
            bin_iv[index] = 0   #异常赋值0（几乎不会出现）

        sum_iv += bin_iv[index]
        total_dict[index] = [bin_lower[index], bin_upper[index], bin_woe[index], bin_iv[index]]

    return total_dict, sum_iv






















