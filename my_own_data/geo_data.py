import math
import torch
import numpy as np
import pandas as pd
from collections import Counter
from torch_geometric.data import Data


def concat_file(cause_files):
    file1 = './train/' + str(cause_files[0]) + '.csv'
    data = pd.read_csv(file1)
    data['label'] = cause_files[0]
    for num in cause_files[1:]:
        file = './train/' + str(num) + '.csv'
        data_me = pd.read_csv(file)
        data_me['label'] = num
        data = data.append(data_me)
    # 删除无关列、全为nan的列
    data.reset_index(drop=True, inplace=True)
    data.drop(columns=['feature3_1', 'feature3_2', 'feature3_3',
                       'feature3_4', 'feature3_5', 'feature3_6',
                       'feature3_7', 'feature3_8', 'feature14', 'Date & Time'],
              inplace=True)
    data.dropna(axis=1, how='all', inplace=True)

    return data


# 按cause类型拼接数据，考虑大于和小于200
def concat_abnormal_file(cause_files):
    file1 = './train/' + str(cause_files[0]) + '.csv'
    data = pd.read_csv(file1)
    data['label'] = cause_files[0]
    data = data[data['feature0'] < 200]
    for num in cause_files[1:]:
        file = './train/' + str(num) + '.csv'
        data_me = pd.read_csv(file)
        data_me['label'] = num
        data_me = data_me[data_me['feature0'] < 200]
        data = data.append(data_me)
    # 删除无关列、全为nan的列
    data.reset_index(drop=True, inplace=True)
    data.drop(columns=['feature3_1', 'feature3_2', 'feature3_3',
                       'feature3_4', 'feature3_5', 'feature3_6',
                       'feature3_7', 'feature3_8', 'feature14', 'Date & Time'],
              inplace=True)
    data.dropna(axis=1, how='all', inplace=True)

    return data


class Data_pre():
    def __init__(self, rows):
        self.rows = rows

    def df2data(self, dataframe, cause_type):
        # dataframe是处理好行数的，rows行
        edge_index = np.load('./cooadj.npy')
        file_label = dataframe['label'].to_numpy()
        dataframe = dataframe.drop(columns='label')

        x = dataframe.T.fillna(0).to_numpy()
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        if x.dtype == 'object':
            print(file_label[0])
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor([cause_type], dtype=torch.long)
        file_label = torch.tensor(file_label, dtype=torch.int)
        data = Data(x=x, edge_index=edge_index, y=y, file_label=file_label)

        return data

    def df_pre(self, dataframe, file):
        dataframe = dataframe[dataframe['feature0'] < 200]
        dataframe.reset_index(drop=True, inplace=True)
        dataframe.drop(columns=['feature3_1', 'feature3_2', 'feature3_3',
                                'feature3_4', 'feature3_5', 'feature3_6',
                                'feature3_7', 'feature3_8', 'feature14',
                                'Date & Time'],
                       inplace=True)
        dataframe.dropna(axis=1, how='all', inplace=True)
        dataframe['label'] = file
        return dataframe

    def df_copy(self, dataframe):
        # 输入不满rows行的数据，补充为rows行
        n = math.ceil(self.rows / dataframe.shape[0])
        dataframe = pd.DataFrame(np.repeat(dataframe.values, n, axis=0),
                                 columns=dataframe.columns)
        return dataframe[:30]

    def file2sample(self, file, cause_type):
        sample_list = []
        if file.shape[0] > 30:
            for num in range(0, file.shape[0], self.rows):
                if file.shape[0] - num < self.rows:
                    break
                df_slice = file[num: num + self.rows]
                sample = self.df2data(df_slice, cause_type)
                sample_list.append(sample)
        elif file.shape[0] < 30:
            file = self.df_copy(file)
            sample = self.df2data(file, cause_type)
            sample_list.append(sample)
        else:
            sample = self.df2data(file, cause_type)
            sample_list.append(sample)
        return sample_list

    def process(self):
        cause_types = ['cause1_nums', 'cause2_nums', 'cause3_nums', 'cause2_3_nums']
        data_list = []
        for i, name in enumerate(cause_types):
            cause_nums = np.load('./data' + '/' + name + '.npy')
            for num in cause_nums:
                if num in [2488, 2752, 2852]:
                    continue
                file = './train/' + str(num) + '.csv'
                data_file = pd.read_csv(file)
                data_file = self.df_pre(data_file, num)
                if data_file.shape[0] == 0:
                    continue
                datas = self.file2sample(data_file, i)
                data_list.extend(datas)
        return data_list


def data_pre():
    rows = 30
    edge_index = np.load('./cooadj.npy')
    cause_type = ['cause1_nums', 'cause2_nums', 'cause3_nums', 'cause2_3_nums']
    data_list = []
    label = []
    for i, name in enumerate(cause_type):
        file_path = './data' + '/' + name + '.npy'
        cause_nums = np.load(file_path)
        df = concat_abnormal_file(cause_nums)

        for num in range(0, df.shape[0], rows):
            if df.shape[0] - num < rows:
                break
            df_slice = df[num: num + rows]
            # save the label column, then drop it
            file_label = df_slice['label'].to_numpy()
            df_slice.drop(columns='label')

            x = df_slice.T.fillna(0).to_numpy()
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            if x.dtype == 'object':
                continue
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor([i], dtype=torch.long)
            file_label = torch.tensor(file_label, dtype=torch.int)

            data = Data(x=x, edge_index=edge_index, y=y, file_label=file_label)
            data_list.append(data)
            label.append(i)
    print(Counter(label))
    return data_list


def test_model(file_path, cause_type):
    edge_index = np.load('./cooadj.npy')
    df = pd.read_csv(file_path)
    if df.shape[0] < 2:
        print('This file has no enough data! ', file_path)
        return None
    # 删除无关列、全为nan的列
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['feature3_1', 'feature3_2', 'feature3_3',
                     'feature3_4', 'feature3_5', 'feature3_6',
                     'feature3_7', 'feature3_8', 'feature14', 'Date & Time'],
            inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    times = df.shape[0] // 50
    data_list = []
    if times >= 1:
        for num in range(0, df.shape[0], 50):
            x = df[num: num + 50].T.fillna(0).to_numpy()
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor([cause_type], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
    else:
        fill_df = pd.DataFrame(np.zeros(shape=(50 - df.shape[0], df.shape[1])), columns=df.columns)
        fill_df.loc[:, :] = 0
        df = df.append(fill_df)

        x = df.T.fillna(0).to_numpy()
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor([cause_type], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list


# if __name__ == '__main__':
#     file_path = './train/' + str(915) + '.csv'
#     data_list = test_model(file_path, 1)
#     print(data_list)
