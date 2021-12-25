import pandas as pd
import numpy as np

def concat_file(cause_files):
    file1 = '../train/' + str(cause_files[0]) + '.csv'
    data = pd.read_csv(file1)
    for num in cause_files[1:]:
        file = '../train/' + str(num) + '.csv'
        data_me = pd.read_csv(file)
        data = data.append(data_me)
    # 删除无关列、全为nan的列
    data.reset_index(drop=True, inplace=True)
    data.drop(columns=['feature3_1', 'feature3_2', 'feature3_3', 
                        'feature3_4', 'feature3_5', 'feature3_6', 
                        'feature3_7', 'feature3_8', 'feature14', 'Date & Time'], 
                        inplace=True)
    data.dropna(axis=1, how='all', inplace=True)

    return data