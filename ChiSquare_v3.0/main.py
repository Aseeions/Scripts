# -*- coding: utf-8 -*-
from chisquare import *
from WOE import *
import pandas as pd

pd.set_option('display.max_columns', None)

# 分箱处理
bins = ChiMerge(temp, 'loan_amnt', 'result', confidenceVal=3.841, bin=5, sample=None)
print('=================================================================')
print('分箱节点列表：\n%s' % bins[1])
print('===========================分箱结果===============================')
print(bins[0])

# WOE/IV计算
a = iv_value('./KS_data.csv', 'loan_amnt', tuple(bins[1][1:-1]), 'result')
print('========================WOE/IV计算结果============================')
print(a[1])
print('=================================================================')
print('总IV值：%0.5f' % a[0])
