from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import to_hetero
from torch_geometric.datasets import IMDB, Actor
from torch_geometric.loader import NeighborLoader
import torch
from torch.nn.functional import dropout
from torch_geometric.nn import to_hetero
from torch_geometric.nn import Linear, SAGEConv

from torch.nn.functional import one_hot

"""Заглавив на error и времето ми истекува за прикачување, грешката неможам да ја поправам и нажалост задачата не ми е 
    завршена, не ми е добар изучен материјалот и веќе незнам што правам 
     Traceback (most recent call last):
  File "D:\\Education\\University\\Courses\\Intelligent Information Systems\\Labs\\Lab 1\\151211_lab1_z2_IIS.py", line 107, in <module>
    train_classification(model, train_loader, val_loader, optimizer, criterion, 1)
  File "D:\\Education\\University\\Courses\\Intelligent Information Systems\\Labs\\Lab 1\\151211_lab1_z2_IIS.py", line 26, in train_classification
    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
IndexError: The shape of the mask [3044, 10] at index 1 does not match the shape of the indexed tensor [3044, 5] at index 1
     """


def train_classification(model, train_loader, val_loader, optimizer, criterion, epochs=5):

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            # if len(batch.train_mask.shape) > 1:
            #     batch.train_mask = batch.train_mask[:, 0]
            # else:
            #     batch.train_mask = dataset.train_mask

            out = model(batch.x, batch.edge_index)
            loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index)
                val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask]).item()
        val_loss /= len(val_loader)
        print(f'Epoch: {epoch + 1}, Val Loss: {val_loss:.4f}')


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = SAGEConv((-1, -1), 64)
        self.conv2 = SAGEConv((-1, -1), 128)
        self.conv3 = SAGEConv((-1, -1), 64)

        self.linear1 = Linear(64, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).tanh()
        x = dropout(x, p=0.3)

        x = self.conv2(x, edge_index).tanh()
        x = dropout(x, p=0.3)

        x = self.conv3(x, edge_index).tanh()
        x = dropout(x, p=0.3)

        x = self.linear1(x)

        return x


if __name__ == '__main__':
    data = Actor('../data')
    dataset = data[0]

    print(dataset)

    model = GraphSAGE(num_classes=5)

    num_nodes = dataset.num_nodes
    indices = list(range(num_nodes))

    #
    train_size = int(num_nodes * 0.5)
    val_size = int(num_nodes * 0.25)
    test_size = int(num_nodes * 0.25)
    train_mask = indices[:train_size]
    val_mask = indices[train_size:train_size + val_size]
    test_mask = indices[train_size + val_size:]

    print(train_mask)
    print(val_mask, test_mask)

    train_input_nodes = torch.tensor(train_mask, dtype=torch.long)
    # train_input_nodes = dataset[train_mask]
    train_loader = NeighborLoader(dataset, num_neighbors=[10, 10, 10],
                                  shuffle=True, input_nodes=train_input_nodes,
                                  batch_size=128)

    val_input_nodes = torch.tensor(val_mask, dtype=torch.long)
    val_loader = NeighborLoader(dataset, num_neighbors=[10, 10, 10],
                                shuffle=False, input_nodes=val_input_nodes,
                                batch_size=128)

    test_input_nodes = torch.tensor(test_mask, dtype=torch.long)
    test_loader = NeighborLoader(dataset, num_neighbors=[10, 10, 10],
                                 shuffle=False, input_nodes=test_input_nodes,
                                 batch_size=128)

    optimizer = SGD(model.parameters(), lr=0.0001)
    criterion = CrossEntropyLoss()


    train_classification(model, train_loader, val_loader, optimizer, criterion, 1)
