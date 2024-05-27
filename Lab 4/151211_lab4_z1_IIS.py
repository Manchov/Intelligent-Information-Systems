import torch
from torch.optim import Adam
from torch_geometric.nn import TransE, ComplEx
from torch_geometric.datasets import FB15k_237


#Фунција за тренирање од kg_model_utils.py
def train(model, data_loader, optimizer, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_examples = 0

        for head_index, rel_type, tail_index in data_loader:
            optimizer.zero_grad()
            loss = model.loss(head_index, rel_type, tail_index)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()

        loss = total_loss / total_examples
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


#Фунции за евалуација од kg_model_utils.py
def evaluate(model, data_loader):
    hits1_list = []
    hits3_list = []
    hits10_list = []
    mr_list = []
    mrr_list = []

    for head_index, rel_type, tail_index in data_loader:
        head_embeds = model.node_emb(head_index)
        relation_embeds = model.rel_emb(rel_type)
        tail_embeds = model.node_emb(tail_index)

        if isinstance(model, TransE):
            scores = torch.norm(head_embeds + relation_embeds - tail_embeds, p=1, dim=1)

        elif isinstance(model, ComplEx):
            # Get real and imaginary parts
            re_relation, im_relation = torch.chunk(relation_embeds, 2, dim=1)
            re_head, im_head = torch.chunk(head_embeds, 2, dim=1)
            re_tail, im_tail = torch.chunk(tail_embeds, 2, dim=1)

            # Compute scores
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            scores = (re_score * re_tail + im_score * im_tail)

            # Negate as we want to rank scores in ascending order, lower the better
            scores = - scores.sum(dim=1)

        else:
            raise ValueError(f'Unsupported model.')

        scores = scores.view(-1, head_embeds.size()[0])

        hits1, hits3, hits10, mr, mrr = eval_metrics(scores)
        hits1_list.append(hits1.item())
        hits3_list.append(hits3.item())
        hits10_list.append(hits10.item())
        mr_list.append(mr.item())
        mrr_list.append(mrr.item())

    hits1 = sum(hits1_list) / len(hits1_list)
    hits3 = sum(hits3_list) / len(hits1_list)
    hits10 = sum(hits10_list) / len(hits1_list)
    mr = sum(mr_list) / len(hits1_list)
    mrr = sum(mrr_list) / len(hits1_list)

    return hits1, hits3, hits10, mr, mrr


def eval_metrics(y_pred):
    argsort = torch.argsort(y_pred, dim=1, descending=False)
    # not using argsort to do the rankings to avoid bias when the scores are equal
    ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
    ranking_list = ranking_list[:, 1] + 1
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    mr_list = ranking_list.to(torch.float)
    mrr_list = 1. / ranking_list.to(torch.float)

    return hits1_list.mean(), hits3_list.mean(), hits10_list.mean(), mr_list.mean(), mrr_list.mean()


if __name__ == "__main__":
    #Одчитување на множеството, исто е се како аудиториската само со заменето податочно множество
    train_data = FB15k_237('../data/FB15k', split='train')[0]
    val_data = FB15k_237('../data/FB15k', split='val')[0]
    test_data = FB15k_237('../data/FB15k', split='test')[0]

    model = TransE(num_nodes=train_data.num_nodes,
                   num_relations=train_data.num_edge_types,
                   hidden_channels=50)

    loader = model.loader(head_index=train_data.edge_index[0],
                          rel_type=train_data.edge_type,
                          tail_index=train_data.edge_index[1],
                          batch_size=1000,
                          shuffle=True)

    optimizer = Adam(model.parameters(), lr=0.01)

    train(model, loader, optimizer)

    hits1, hits3, hits10, mr, mrr = evaluate(model, loader)

    print(f'hits@1: {hits1: .3f}, hits@3: {hits3: .3f}, hits@10: {hits10: .3f}, и MRR: {mrr: .3f}')
    """ Output:
    Epoch: 000, Loss: 0.7592
    Epoch: 001, Loss: 0.5682
    Epoch: 002, Loss: 0.4556
    Epoch: 003, Loss: 0.3665
    Epoch: 004, Loss: 0.3044
    Epoch: 005, Loss: 0.2656
    Epoch: 006, Loss: 0.2426
    Epoch: 007, Loss: 0.2254
    Epoch: 008, Loss: 0.2146
    Epoch: 009, Loss: 0.2039
    Epoch: 010, Loss: 0.1957
    Epoch: 011, Loss: 0.1888
    Epoch: 012, Loss: 0.1832
    Epoch: 013, Loss: 0.1778
    Epoch: 014, Loss: 0.1740
    Epoch: 015, Loss: 0.1707
    Epoch: 016, Loss: 0.1650
    Epoch: 017, Loss: 0.1622
    Epoch: 018, Loss: 0.1602
    Epoch: 019, Loss: 0.1581
    Epoch: 020, Loss: 0.1553
    Epoch: 021, Loss: 0.1518
    Epoch: 022, Loss: 0.1490
    Epoch: 023, Loss: 0.1483
    Epoch: 024, Loss: 0.1455
    Epoch: 025, Loss: 0.1449
    Epoch: 026, Loss: 0.1431
    Epoch: 027, Loss: 0.1423
    Epoch: 028, Loss: 0.1412
    Epoch: 029, Loss: 0.1378
    Epoch: 030, Loss: 0.1375
    Epoch: 031, Loss: 0.1354
    Epoch: 032, Loss: 0.1345
    Epoch: 033, Loss: 0.1340
    Epoch: 034, Loss: 0.1316
    Epoch: 035, Loss: 0.1308
    Epoch: 036, Loss: 0.1287
    Epoch: 037, Loss: 0.1281
    Epoch: 038, Loss: 0.1263
    Epoch: 039, Loss: 0.1260
    Epoch: 040, Loss: 0.1257
    Epoch: 041, Loss: 0.1233
    Epoch: 042, Loss: 0.1230
    Epoch: 043, Loss: 0.1215
    Epoch: 044, Loss: 0.1207
    Epoch: 045, Loss: 0.1197
    Epoch: 046, Loss: 0.1188
    Epoch: 047, Loss: 0.1184
    Epoch: 048, Loss: 0.1171
    Epoch: 049, Loss: 0.1169
    Epoch: 050, Loss: 0.1160
    Epoch: 051, Loss: 0.1150
    Epoch: 052, Loss: 0.1148
    Epoch: 053, Loss: 0.1152
    Epoch: 054, Loss: 0.1128
    Epoch: 055, Loss: 0.1124
    Epoch: 056, Loss: 0.1120
    Epoch: 057, Loss: 0.1122
    Epoch: 058, Loss: 0.1107
    Epoch: 059, Loss: 0.1105
    Epoch: 060, Loss: 0.1102
    Epoch: 061, Loss: 0.1098
    Epoch: 062, Loss: 0.1088
    Epoch: 063, Loss: 0.1089
    Epoch: 064, Loss: 0.1066
    Epoch: 065, Loss: 0.1073
    Epoch: 066, Loss: 0.1084
    Epoch: 067, Loss: 0.1064
    Epoch: 068, Loss: 0.1054
    Epoch: 069, Loss: 0.1047
    Epoch: 070, Loss: 0.1044
    Epoch: 071, Loss: 0.1039
    Epoch: 072, Loss: 0.1039
    Epoch: 073, Loss: 0.1047
    Epoch: 074, Loss: 0.1036
    Epoch: 075, Loss: 0.1026
    Epoch: 076, Loss: 0.1031
    Epoch: 077, Loss: 0.1021
    Epoch: 078, Loss: 0.1018
    Epoch: 079, Loss: 0.1026
    Epoch: 080, Loss: 0.1013
    Epoch: 081, Loss: 0.1012
    Epoch: 082, Loss: 0.1019
    Epoch: 083, Loss: 0.1003
    Epoch: 084, Loss: 0.1010
    Epoch: 085, Loss: 0.1012
    Epoch: 086, Loss: 0.1007
    Epoch: 087, Loss: 0.0999
    Epoch: 088, Loss: 0.0997
    Epoch: 089, Loss: 0.0996
    Epoch: 090, Loss: 0.0995
    Epoch: 091, Loss: 0.0990
    Epoch: 092, Loss: 0.0986
    Epoch: 093, Loss: 0.0985
    Epoch: 094, Loss: 0.0983
    Epoch: 095, Loss: 0.0983
    Epoch: 096, Loss: 0.0982
    Epoch: 097, Loss: 0.0974
    Epoch: 098, Loss: 0.0972
    Epoch: 099, Loss: 0.0971
    hits@1:  0.000, hits@3:  0.015, hits@10:  0.015, и MRR:  0.009
    """