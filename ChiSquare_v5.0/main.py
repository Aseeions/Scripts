# -*- coding: utf-8 -*-
from chisq_bin_cate import chisq_bin_category
from chisq_bin import chisq_bin_continuous
import pandas as pd



if __name__ == '__main__':

    data = pd.read_csv('./KS_data.csv', sep=',')

    res_category = chisq_bin_category(df=data, feature='type', target='result', n_bins=5, min_samples=None, chi_threshold=None, balance=True)

    res_continuous = chisq_bin_continuous(df=data, feature='loan_amnt', target='result', n_bins=5, min_samples=None, chi_threshold=None, balance=True)

    for bin in res_category:
        print(bin)
    print('================')
    print(res_continuous)
