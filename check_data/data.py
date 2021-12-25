import pandas as pd
import numpy as np


class Data():
    def __init__(self, 
                train_label_path, 
                cause_type):
        self.cause_type = cause_type
        self.train_label_path = train_label_path

    def cause_type_data_list(self):
        # 获取某种根因下的所有数据列表
        train_label = pd.read_csv(self.train_label_path)
        cause_nums = train_label[train_label['root-cause(s)']==self.cause_type]['sample_index'].to_list()
        return cause_nums

    def corr_of_features(self, file_num, feature_name1, feature_name2):
        # 返回两个指标间的相关性
        file_path = './train/' + str(file_num) + '.csv'
        df = pd.read_csv(file_path)
        columns = df.columns.to_list()
        zero_columns = df.loc[:, (df==0).any()].columns.tolist()
        for zero_column in zero_columns:
            columns.remove(zero_column)
        corr = df[feature_name1].corr(df[feature_name2])
        return corr

    def get_corr_top(cause_nums, top_k=5):
        # cause_nums：根因为n的样本列表
        # all_top_lists：[{'456': ['feature2',...,]}, ..., ]
        all_top_lists = []
        for num in cause_nums:
            file_name = str(num) + '.csv'
            file_path = './train/' + file_name
            one_top = get_top(file_path, top_k)
            all_top_lists.append({str(num): one_top})
        return all_top_lists
