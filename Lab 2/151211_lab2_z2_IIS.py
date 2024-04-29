import torch

import pandas as pd
from torch_geometric.nn import SAGEConv, to_hetero, Linear
from torch_geometric.datasets import AmazonBook
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from torch.nn.functional import mse_loss
from gnn_link_prediction import Model, train_link_prediction
from torch.optim import SGD
import networkx.algorithms.tests.test_link_prediction as test_link_prediction
import os.path as osp
from torch_geometric.data import HeteroData




if __name__ == '__main__':
    # Иако имам торч со CUDA 11.8 инсталирано, на мојот уред сеуште не е подесено CUDA но за во иднина да ми служи со Графичката без да модифицирам код
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Amazon')
    # # dataset = AmazonBook(path)
    dataset = AmazonBook('../data/amazon-book')
    data = dataset[0]
    print(dataset.data)
    print(data)
    print(data['user'])  # To see if 'user' is in the data dictionary
    print(data['book'])  # To see if 'book' is in the data dictionary
    num_users, num_books = data['user'].num_nodes, data['book'].num_nodes
    data['user'].x = torch.ones(num_users, 1)
    data['book'].x = torch.ones(num_books, 1)
    # data['user'].x = torch.ones(num_users, 1)
    # data['book'].x = torch.ones(num_books, 1)
    # # data = data.to_homogeneous().to(device)
    #
    # # # Лоадирање на податочното множество
    # # dataset = AmazonBook('../data')
    # # data = dataset[0]
    # #
    # # num_users = dataset["user"].num_nodes
    # # num_books = dataset["book"].num_nodes

    #
    train_val_test_split = RandomLinkSplit(num_val=0.2,
                                           num_test=0.2,
                                           add_negative_train_samples=True,
                                           edge_types=('user', 'rates', 'book'),
                                           rev_edge_types=('book', 'rated_by', 'user'))


    train_data, val_data, test_data = train_val_test_split(data)

    model = Model(hidden_channels=128, data=data)

    optimizer = SGD(model.parameters(), lr=0.0001)

    train_link_prediction(model, train_data, val_data, optimizer, 100)
    test_link_prediction(model, test_data, optimizer, 100)
