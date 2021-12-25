import networkx as nx
import numpy as np
from numpy.lib import type_check
import pandas as pd
import matplotlib.pyplot as plt
from cdt.causality.pairwise import ANM, BivariateFit, IGCI
from networkx.algorithms.centrality.degree_alg import out_degree_centrality
from cdt.data import load_dataset
from tools import *


if __name__ == '__main__':

    cause1_nums = np.load('cause1_nums.npy')
    type_result = {}
    for num in cause1_nums:
        file_path = './train/' + str(num) + '.csv'
        data = pd.read_csv(file_path)
        if data.shape[0] < 10:
            continue
        data = data.drop(columns='Date & Time').dropna(axis=1, how='all').fillna(0)
        
        columns = data.columns.to_list()
        columns.remove('feature0')
        data = df2pair(data, 1, columns)
        
        obj = IGCI()
        
        output = obj.predict(data)
        
        print(num)

        type_result.update({num: top_features(output, columns)[:5]})


    print(type_result)
# kara_pos = nx.spring_layout(output, k=0.5)
# nx.draw_networkx(output, 
#                 kara_pos, 
#                 font_size=8)

# plt.show()