import networkx as nx
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.datasets import AmazonBook
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_networkx

if __name__ == '__main__':

    # Иако имам торч со CUDA 11.8 инсталирано, на мојот уред сеуште не е подесено CUDA но за во иднина да ми служи со Графичката без да модифицирам код
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Лоадирање на податочното множество
    dataset = AmazonBook('../data')
    data = dataset[0]

    """ Креирање на моделот за трансформација со RandomLinkSplit,со соодветни барањата на задачата
    - num_test = 20% од врските како позитивни примероци
    - neg_sampling_ratio = толку врски во графот кои не постојат и ќе претставуваат негативни примероци при тестирањето
    """
    transform = RandomLinkSplit(num_val=0.1,
                                num_test=0.2,
                                is_undirected=True,
                                neg_sampling_ratio=1.0,
                                split_labels=True)

    # Делба со примена на трансформацјата
    train_data, val_data, test_data = transform(data)


    """Претварање на множеството во граф за да може да работи со networkx,
    каде се користе jaccard_coefficient методот за главната пресметка. Овде највише заглаив бидејки не го разбирав податочниот модел"""
    data_graph = to_networkx(train_data, to_undirected=True)

    # Вадење на позитивни и негативни рабови
    test_pos_edge_index = test_data.pos_edge_label_index
    test_neg_edge_index = test_data.neg_edge_label_index

    jaccard_scores = []
    labels = []

    """За да се пресмета Jaccard Coefficient за позитивни и негативни рабови, ги бележиме со 
        (1 - позитив/постиои), (0 - негатив/непостои) за да знаеме од кој раб каков е
        потоа засекоја јазла у реброто ја мереме сличноста на соседните јазли,
        резултатот се добива како трет елемент во торката што ја враќа функцијата, затоа се користе индексот [0][2] 
        и на крај се додаваат на листата на резултати и неговиот тип во листа лабелата
    """
    for edge_index, label in zip([test_pos_edge_index, test_neg_edge_index], [1, 0]):
        for u, v in edge_index.t().numpy():
            score = list(nx.jaccard_coefficient(data_graph, [(u, v)]))[0][2]
            jaccard_scores.append(score)
            labels.append(label)

    # ги пресметуваме ROC AUC and Average Precision score со соодветните параметри и ги печати
    roc_auc = roc_auc_score(labels, jaccard_scores)
    average_precision = average_precision_score(labels, jaccard_scores)

    print(f"ROC AUC Score: {roc_auc}")
    print(f"Average Precision Score: {average_precision}")

    """Резултат/пречатење:
        ROC AUC Score: 0.573712557357825
        Average Precision Score: 0.5701004655127894
    """
