import os
import torch
from geo_data import Data_pre
from torch_geometric.data import InMemoryDataset, download_url


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        file_names = os.listdir(self.raw_dir)
        # file_names.remove('README.txt')
        # file_names.remove('README.txt~')
        return file_names

    @property
    def processed_file_names(self):
        return ['data.pt']

    # def download(self):
    #     download_url(url, self.raw_dir)
    def process(self):
        # Read data into huge `Data` list.
        data_pre = Data_pre(30)
        data_list = data_pre.process()
        # data_list = data_pre()

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == '__main__':
    test = MyOwnDataset(root='./')

    # data_list = data_pre()