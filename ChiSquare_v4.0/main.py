import  pandas as pd
from woe_cal import  woe_cal
from woe_replace import woe_replace
from chisq_bin import chi_square_bins


data = pd.read_csv('./KS_data.csv', sep=',')

if __name__ == '__main__':

    cutNodes = chi_square_bins(data, 'loan_amnt', 'result', bin = 5)

    result,iv = woe_cal(data, 'loan_amnt', 'result', cutNodes[1:-1])
    print(result)
    print(iv)

    # 替换
    for item in result.items():
        data = woe_replace(data, 'loan_amnt', result)
    print(data)











