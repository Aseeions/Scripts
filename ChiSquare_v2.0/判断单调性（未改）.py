# -*- coding: utf-8 -*-
import pandas as pd


#原文：https://blog.csdn.net/yaoqsm/article/details/83828565 


#单调性，计算每个箱中，坏样本占本箱内所有样本的比例
def BadRateMonotone(df, sortByVar, target):
    '''
    参数：数据，已分箱的列（字符串），标签列（字符串）
    返回值：True/False
    '''
    #df[sortByVar]这列已经经过分箱
    df2=df.sort_values(by=[sortByVar])
    total=df2.groupby([sortByVar])[target].count()
    total1=pd.DataFrame({'total':total})
    print(total1)
    good=df2.groupby([sortByVar])[target].sum()
    #good1=pd.DataFrame({'good':good})
    bad=total-good
    bad=pd.DataFrame({'bad':bad})
    regroup=total1.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    combined=zip(regroup['total'], regroup['bad'])
    badRate=[x[1]*1.0/x[0] for x in combined]
    badRateMonotone=[badRate[i]>badRate[i+1] for i in range(len(badRate)-1)]
    Monotone = len(set(badRateMonotone))
    if Monotone==1:
        return True
    else:
        return False
