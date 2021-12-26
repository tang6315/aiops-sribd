import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

model = Model(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


model.eval()
correct = 0.0
loss_test = 0.0
data_list = []

for data in data_list:
    data = data.to(args.device)
    out = model(data)
    pred = out.max(dim=1)[1]
    correct += pred.eq(data.y).sum().item()
    loss_test += F.nll_loss(out, data.y).item()



