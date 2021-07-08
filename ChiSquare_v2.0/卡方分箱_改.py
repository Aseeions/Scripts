# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
#data = pd.read_csv('iris.csv', sep="\t", na_values=['', '?'])
#temp = data[['x','y']]
data=pd.read_csv('./KS_data.csv')
temp=data[['loan_amnt','result']]


 
# 定义一个卡方分箱（可设置参数置信度水平与箱的个数）停止条件为大于置信水平且小于bin的数目
def ChiMerge(df, variable, flag, confidenceVal=3.841, bin=10, sample = None):
#    return df
    '''
    #运行前需要 import pandas as pd 和 import numpy as np
    #df:传入一个数据框仅包含一个需要卡方分箱的变量与正负样本标识（正样本为1，负样本为0）
    #variable:需要卡方分箱的变量名称（字符串）
    #flag：正负样本标识的名称（字符串）
    #confidenceVal：置信度水平（默认是不进行抽样95%）
    #bin：最多箱的数目
    #sample: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
    '''	
#统计所用个体数，计算期望值
    total_count=df[variable].count()
    total_positive_class=df[flag].sum()
    total_positive_rate=total_positive_class/total_count
    total_negative_rate=1-total_positive_rate
    
    
#进行数据格式化录入、
    total_num = df.groupby([variable])[flag].count()  # 统计需分箱变量每个值数目
    total_num = pd.DataFrame({'total_num': total_num})  # 创建一个数据框保存之前的结果
    positive_class = df.groupby([variable])[flag].sum()  # 统计需分箱变量每个值正样本数
    positive_class = pd.DataFrame({'positive_class': positive_class})  # 创建一个数据框保存之前的结果NO.1
    regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True,
                       how='inner')  # 组合total_num与positive_class
    regroup.reset_index(inplace=True)
    regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  # 统计需分箱变量每个值负样本数NO.2
    regroup['positive_expect']=regroup['total_num']*total_positive_rate  #统计需分箱变量每个值期望的负样本数NO.3
    regroup['negative_expect']=regroup['total_num']*total_negative_rate  #统计需分箱变量每个值期望的正样本数NO.4
    regroup = regroup.drop('total_num', axis=1) #特征/正样本数/负样本数
    np_regroup = np.array(regroup)  # 把数据框转化为numpy（提高运行效率）
    print('已完成数据读入,正在计算数据初处理...')


#处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
    i = 0
    while (i <= np_regroup.shape[0] - 2):
        if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) 
        or ( np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
            np_regroup[i,3]=np_regroup[i,3]+np_regroup[i+1,3]   #期望的正样本
            np_regroup[i,4]=np_regroup[i,4]+np_regroup[i+1,4]   #期望的负样本
            np_regroup[i, 0] = np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i = i - 1
        i = i + 1
 
    
#对相邻两个区间进行卡方值计算
    chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
    for i in np.arange(np_regroup.shape[0] - 1):
        chi=(np_regroup[i,1]-np_regroup[i,3])**2/np_regroup[i,3]\
            +(np_regroup[i+1,1]-np_regroup[i+1,3])**2/np_regroup[i+1,3]\
            +(np_regroup[i,2]-np_regroup[i,4])**2/np_regroup[i,4]\
            +(np_regroup[i+1,2]-np_regroup[i+1,4])**2/np_regroup[i+1,4]
        chi_table = np.append(chi_table, chi)
    print('已完成数据初处理，正在进行卡方分箱核心操作...')


#把卡方值最小的两个区间进行合并（卡方分箱核心）
    while (1):
        if (len(chi_table) <= (bin - 1) and min(chi_table) >= confidenceVal):
            break
        chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  # 找出卡方值最小的位置索引
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]  #正样本合并
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]  #负样本合并
        np_regroup[chi_min_index, 3] = np_regroup[chi_min_index, 3] + np_regroup[chi_min_index + 1, 3]  #期望的正样本
        np_regroup[chi_min_index, 4] = np_regroup[chi_min_index, 4] + np_regroup[chi_min_index + 1, 4]  #期望的负样本
        np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
        np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

        if (chi_min_index == np_regroup.shape[0] - 1):  # 最小值是最后两个区间的时候
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] =(np_regroup[chi_min_index-1,1]-np_regroup[chi_min_index-1,3])**2/np_regroup[chi_min_index-1,3]\
                                            +(np_regroup[chi_min_index,1]-np_regroup[chi_min_index,3])**2/np_regroup[chi_min_index,3]\
                                            +(np_regroup[chi_min_index-1,2]-np_regroup[chi_min_index-1,4])**2/np_regroup[chi_min_index-1,4]\
                                            +(np_regroup[chi_min_index,2]-np_regroup[chi_min_index,4])**2/np_regroup[chi_min_index,4]
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index, axis=0)
        #-------------------------------------------------------------------------
        elif (chi_min_index == 0):  #最小值是前两个区间的时候
            chi_table[chi_min_index] =(np_regroup[chi_min_index,1]-np_regroup[chi_min_index,3])**2/np_regroup[chi_min_index,3]\
                                            +(np_regroup[chi_min_index+1,1]-np_regroup[chi_min_index+1,3])**2/np_regroup[chi_min_index+1,3]\
                                            +(np_regroup[chi_min_index,2]-np_regroup[chi_min_index,4])**2/np_regroup[chi_min_index,4]\
                                            +(np_regroup[chi_min_index+1,2]-np_regroup[chi_min_index+1,4])**2/np_regroup[chi_min_index+1,4]
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index+1, axis=0)
        #-------------------------------------------------------------------------
        else:
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] =(np_regroup[chi_min_index-1,1]-np_regroup[chi_min_index-1,3])**2/np_regroup[chi_min_index-1,3]\
                                            +(np_regroup[chi_min_index,1]-np_regroup[chi_min_index,3])**2/np_regroup[chi_min_index,3]\
                                            +(np_regroup[chi_min_index-1,2]-np_regroup[chi_min_index-1,4])**2/np_regroup[chi_min_index-1,4]\
                                            +(np_regroup[chi_min_index,2]-np_regroup[chi_min_index,4])**2/np_regroup[chi_min_index,4]
            # 计算合并后当前区间与后一个区间的卡方值并替换
            chi_table[chi_min_index] =(np_regroup[chi_min_index,1]-np_regroup[chi_min_index,3])**2/np_regroup[chi_min_index,3]\
                                            +(np_regroup[chi_min_index+1,1]-np_regroup[chi_min_index+1,3])**2/np_regroup[chi_min_index+1,3]\
                                            +(np_regroup[chi_min_index,2]-np_regroup[chi_min_index,4])**2/np_regroup[chi_min_index,4]\
                                            +(np_regroup[chi_min_index+1,2]-np_regroup[chi_min_index+1,4])**2/np_regroup[chi_min_index+1,4]
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
    print('已完成卡方分箱核心操作，正在保存结果...')


#把结果保存成一个数据框
    result_data = pd.DataFrame()  # 创建一个保存结果的数据框
    result_data['variable'] = [variable] * np_regroup.shape[0]  # 结果表第一列：变量名
    list_temp = []
    for i in np.arange(np_regroup.shape[0]):
        if i == 0:
            x = '0' + ',' + str(np_regroup[i, 0])
        elif i == np_regroup.shape[0] - 1:
            x = str(np_regroup[i - 1, 0]) + '+'
        else:
            x = str(np_regroup[i - 1, 0]) + ',' + str(np_regroup[i, 0])
        list_temp.append(x)
    result_data['interval'] = list_temp       # 结果表第二列：区间
    result_data['flag_0'] = np_regroup[:, 2]  # 结果表第三列：负样本数目
    result_data['flag_1'] = np_regroup[:, 1]  # 结果表第四列：正样本数目
    result_data['expe_0']=np_regroup[:,4]   # # 结果表第五列：期望负样本数目
    result_data['expe_1']=np_regroup[:,3]    ## 结果表第六列：期望正样本数目
    #切割点例表
    cutNodes=['-inf']+list(np_regroup[:-1,0])+['inf']
    
    return result_data,cutNodes


#调用函数参数示例
#bins = ChiMerge(temp, 'loan_amnt','result', confidenceVal=3.841, bin=4,sample=None)
#print(bins[0])
#print('============================================')
#print(bins[1])















