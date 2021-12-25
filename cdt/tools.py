import pandas as pd
import numpy as np

def df2pair(df, cause_type, features):
    if cause_type == 1:
        data_pair = pd.DataFrame(columns=['left', 'right'])
        for feature in features:
            # if feature == 'feature0':
            #     continue
            new_row = {'left': df['feature0'].to_numpy(), 'right': df[feature].to_numpy()}
            series = pd.Series(new_row, name='0' + '_' + feature[7:])
            data_pair = data_pair.append(series)
    else:
        pass
    return data_pair

def top_features(cdt_result, features):
    dict = {}
    for num, feature in enumerate(features):
        # if feature == 'feature0':
        #     continue
        dict.update({feature: cdt_result[num]})
    return sorted(dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

