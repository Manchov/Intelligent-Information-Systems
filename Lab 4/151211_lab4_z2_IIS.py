import torch
from torch.optim import Adam
from torch_geometric.nn import TransE, ComplEx
from torch_geometric.datasets import FB15k_237


# Фунција за тренирање од kg_model_utils.py
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


# Фунции за евалуација од kg_model_utils.py
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
    # Одчитување на множеството, исто е се како аудиториската само со заменето податочно множество
    train_data = FB15k_237('../data/FB15k', split='train')[0]
    val_data = FB15k_237('../data/FB15k', split='val')[0]
    test_data = FB15k_237('../data/FB15k', split='test')[0]

    model = ComplEx(num_nodes=train_data.num_nodes,
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
    """ComplEx моделот покажува подобри резултати у споредба со TransE, поготово во послабите вредности на загуба и 
    подобрите евалуациски метрики.
    
    Output:
    Epoch: 000, Loss: 0.5515
    Epoch: 001, Loss: 0.3113
    Epoch: 002, Loss: 0.1766
    Epoch: 003, Loss: 0.1311
    Epoch: 004, Loss: 0.1119
    Epoch: 005, Loss: 0.1019
    Epoch: 006, Loss: 0.0938
    Epoch: 007, Loss: 0.0899
    Epoch: 008, Loss: 0.0855
    Epoch: 009, Loss: 0.0815
    Epoch: 010, Loss: 0.0791
    Epoch: 011, Loss: 0.0783
    Epoch: 012, Loss: 0.0766
    Epoch: 013, Loss: 0.0748
    Epoch: 014, Loss: 0.0746
    Epoch: 015, Loss: 0.0734
    Epoch: 016, Loss: 0.0720
    Epoch: 017, Loss: 0.0705
    Epoch: 018, Loss: 0.0684
    Epoch: 019, Loss: 0.0673
    Epoch: 020, Loss: 0.0693
    Epoch: 021, Loss: 0.0682
    Epoch: 022, Loss: 0.0673
    Epoch: 023, Loss: 0.0666
    Epoch: 024, Loss: 0.0655
    Epoch: 025, Loss: 0.0664
    Epoch: 026, Loss: 0.0641
    Epoch: 027, Loss: 0.0641
    Epoch: 028, Loss: 0.0640
    Epoch: 029, Loss: 0.0645
    Epoch: 030, Loss: 0.0622
    Epoch: 031, Loss: 0.0632
    Epoch: 032, Loss: 0.0638
    Epoch: 033, Loss: 0.0624
    Epoch: 034, Loss: 0.0609
    Epoch: 035, Loss: 0.0626
    Epoch: 036, Loss: 0.0615
    Epoch: 037, Loss: 0.0611
    Epoch: 038, Loss: 0.0603
    Epoch: 039, Loss: 0.0602
    Epoch: 040, Loss: 0.0601
    Epoch: 041, Loss: 0.0599
    Epoch: 042, Loss: 0.0590
    Epoch: 043, Loss: 0.0596
    Epoch: 044, Loss: 0.0591
    Epoch: 045, Loss: 0.0587
    Epoch: 046, Loss: 0.0587
    Epoch: 047, Loss: 0.0602
    Epoch: 048, Loss: 0.0602
    Epoch: 049, Loss: 0.0599
    Epoch: 050, Loss: 0.0584
    Epoch: 051, Loss: 0.0594
    Epoch: 052, Loss: 0.0587
    Epoch: 053, Loss: 0.0578
    Epoch: 054, Loss: 0.0580
    Epoch: 055, Loss: 0.0578
    Epoch: 056, Loss: 0.0577
    Epoch: 057, Loss: 0.0569
    Epoch: 058, Loss: 0.0573
    Epoch: 059, Loss: 0.0568
    Epoch: 060, Loss: 0.0561
    Epoch: 061, Loss: 0.0556
    Epoch: 062, Loss: 0.0578
    Epoch: 063, Loss: 0.0582
    Epoch: 064, Loss: 0.0580
    Epoch: 065, Loss: 0.0576
    Epoch: 066, Loss: 0.0578
    Epoch: 067, Loss: 0.0579
    Epoch: 068, Loss: 0.0567
    Epoch: 069, Loss: 0.0567
    Epoch: 070, Loss: 0.0566
    Epoch: 071, Loss: 0.0577
    Epoch: 072, Loss: 0.0566
    Epoch: 073, Loss: 0.0572
    Epoch: 074, Loss: 0.0579
    Epoch: 075, Loss: 0.0580
    Epoch: 076, Loss: 0.0556
    Epoch: 077, Loss: 0.0571
    Epoch: 078, Loss: 0.0566
    Epoch: 079, Loss: 0.0569
    Epoch: 080, Loss: 0.0557
    Epoch: 081, Loss: 0.0569
    Epoch: 082, Loss: 0.0555
    Epoch: 083, Loss: 0.0562
    Epoch: 084, Loss: 0.0576
    Epoch: 085, Loss: 0.0570
    Epoch: 086, Loss: 0.0579
    Epoch: 087, Loss: 0.0554
    Epoch: 088, Loss: 0.0552
    Epoch: 089, Loss: 0.0570
    Epoch: 090, Loss: 0.0572
    Epoch: 091, Loss: 0.0571
    Epoch: 092, Loss: 0.0558
    Epoch: 093, Loss: 0.0577
    Epoch: 094, Loss: 0.0567
    Epoch: 095, Loss: 0.0566
    Epoch: 096, Loss: 0.0572
    Epoch: 097, Loss: 0.0578
    Epoch: 098, Loss: 0.0561
    Epoch: 099, Loss: 0.0568
    hits@1:  0.004, hits@3:  0.011, hits@10:  0.022, и MRR:  0.014
    """
