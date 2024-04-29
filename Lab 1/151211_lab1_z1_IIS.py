import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from torch.optim import SparseAdam
from torch_geometric.datasets import Actor
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

#Фунција за изцртување на графот
def draw_graph(node_embeddings):
    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(node_embeddings)

    plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1],
                c=labels, cmap='jet', alpha=0.7)
    plt.show()


# node_embeddings.py копирано од првата аудиториска без никакви модификации
def train(model, epochs=5, batch_size=32, lr=0.01, device='cpu'):
    model = model.to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = SparseAdam(list(model.parameters()), lr=lr)

    model.train()

    for epoch in range(epochs):
        train_loss = 0

        for pos_rw, neg_rw in tqdm(loader):
            optimizer.zero_grad()

            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(loader)

        print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}')



if __name__ == "__main__":
    # Лоадирање на податочното множество
    data = Actor('../dataset')
    dataset = data[0]

    # Иако имам торч со CUDA 11.8 инсталирано, на мојот уред сеуште не е подесено CUDA но за во иднина да ми служи со Графичката без да модифицирам код
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Креирање на моделот со Node2Vec
    model = Node2Vec(dataset.edge_index,
                     embedding_dim=50,
                     walk_length=30,
                     context_size=10,
                     walks_per_node=20,
                     num_negative_samples=1,
                     p=200, q=1,
                     sparse=True)

    #Тренирање на моделот со соодветниот процесот
    train(model, epochs=5, device=device.type)

    # Подготовка на податоците за класификаци
    labels = dataset.y.detach().cpu().numpy()
    node_embeddings = model().detach().cpu().numpy()

    # Делење на податоците за тренирање и тестирање
    train_x, test_x, train_y, test_y = train_test_split(node_embeddings, labels,
                                                        test_size=0.1,
                                                        stratify=labels)

    #LogisticRegression најдобри резилтати извади у споредба со RandomForestClassifier и DecisionTreeClassifier
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(train_x, train_y)

    #Пресметки на потребните резултати
    predictions = classifier.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    precision = precision_score(test_y, predictions, average='macro')
    recall = recall_score(test_y, predictions, average='macro')
    f1 = f1_score(test_y, predictions, average='macro')

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

    # Изтртување на графот, за барањаа на вежбава не е потребно
    # draw_graph(node_embeddings)

    """
    Program output:
    100%|██████████| 238/238 [00:07<00:00, 31.43it/s]
    Epoch: 00, Loss: 2.1268
    100%|██████████| 238/238 [00:07<00:00, 33.89it/s]
    Epoch: 01, Loss: 0.9467
    100%|██████████| 238/238 [00:06<00:00, 34.10it/s]
    Epoch: 02, Loss: 0.8573
    100%|██████████| 238/238 [00:06<00:00, 34.03it/s]
    Epoch: 03, Loss: 0.8370
    100%|██████████| 238/238 [00:06<00:00, 34.03it/s]
    Epoch: 04, Loss: 0.8285
    Accuracy: 0.25394736842105264, Precision: 0.23252852201533666, Recall: 0.207772949896151, F1 Score: 0.17044087132454905
    ---
    Резултати од RandomForestClassifier и DecisionTreeClassifier соодветно:
    Accuracy: 0.23421052631578948, Precision: 0.17396872911578795, Recall: 0.1993476554970139, F1 Score: 0.1823566674230391
    Accuracy: 0.23289473684210527, Precision: 0.2165889597371026, Recall: 0.2156520873230643, F1 Score: 0.21594807427104118
    """