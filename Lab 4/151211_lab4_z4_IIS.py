from torch.optim import Adam
from pykeen import predict
from pykeen.models import TransE
from pykeen.datasets import FB15k237, Nations
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator

""" Функција за предвидување и проверка за сите релации, бидејки претпоставувам треба за сите релации да се преверат 
предвидувањата ова фукција за сите ги излистува, но резултатите во споредба со вистинските вредности не ми излегуваат 
како што треба што претпоставувам дека го имам многу утнато моделот или предвидувањето, пр:

Предвидени релации за "usa" со "weightedunvote" релацијата

TargetPredictions(df=    tail_id      score      tail_label
                    12       12      -9.643640          usa
                    11       11     -10.022515           uk
                    5         5     -10.272678        india
                    4         4     -10.344516        egypt
                    9         9     -10.404998  netherlands
                    7         7     -10.578695       israel
                    10       10     -10.635209       poland
                    0         0     -10.842403       brazil
                    1         1     -10.867046        burma
                    8         8     -10.995027       jordan
                    6         6     -11.073771    indonesia
                    13       13     -11.106088         ussr
                    3         3     -11.259303         cuba
                    2         2     -11.428341        china, ... 

Вистински релации за "usa" со "weightedunvote" релацијата:
['cuba', 'israel', 'ussr']

Помала е веројатноста за можноста за дадените вредности у споредбата со вистинската вредност а и резултатите 
од евалуцијата на моделот ми кажува дека не ми дава позитивни резулатите иако @10 е речиси 1 ама од 14 ентитети
hits@1:  0.045, hits@3:  0.664, hits@10:  0.980, и MRR:  0.383

"""


def predict_and_compare(entity, dataset, model):
    print(f'Предвидувања за "{entity}":')
    for rel_id, rel_label in dataset.testing.relation_id_to_label.items():
        rel_label = rel_label.strip()
        preds = predict.predict_target(model=model,
                                       head=entity,
                                       relation=rel_label,
                                       triples_factory=dataset.testing)
        print(f'\nПредвидени релации за "{entity}" со "{rel_label}" релацијата\n')
        print(preds)

        true_triples = dataset.testing.mapped_triples[
            (dataset.testing.mapped_triples[:, 0] == dataset.entity_to_id[entity]) &
            (dataset.testing.mapped_triples[:, 1] == rel_id)]
        true_entities = [dataset.testing.entity_id_to_label[tail_id.item()] for tail_id in true_triples[:, 2]]
        print(f'\nВистински релации за "{entity}" со "{rel_label}" релацијата:')
        print(true_entities)
        print(f'\n')
        print('─' * 100)


if __name__ == '__main__':
    dataset = Nations()

    model = TransE(triples_factory=dataset.training)
    optimizer = Adam(params=model.get_grad_params())
    trainer = SLCWATrainingLoop(model=model,
                                triples_factory=dataset.training,
                                optimizer=optimizer)

    trainer.train(triples_factory=dataset.training,
                  num_epochs=100,
                  batch_size=64)

    evaluator = RankBasedEvaluator()
    res = evaluator.evaluate(model=model,
                             mapped_triples=dataset.testing.mapped_triples,
                             batch_size=128,
                             additional_filter_triples=[dataset.training.mapped_triples,
                                                        dataset.validation.mapped_triples])

    predict_and_compare("uk", dataset, model)

    predict_and_compare("usa", dataset, model)

    hits1 = res.get_metric('hits_at_1')
    hits3 = res.get_metric('hits_at_3')
    hits10 = res.get_metric('hits_at_10')
    mrr = res.get_metric('mean_reciprocal_rank')
    print(f'hits@1: {hits1: .3f}, hits@3: {hits3: .3f}, hits@10: {hits10: .3f}, и MRR: {mrr: .3f}')

    """
    	C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\python.exe "D:\\Education\\University\\Courses\\Intelligent Information Systems\\Labs\\Lab 4\\151211_lab4_z4_IIS.py" 
	No random seed is specified. This may lead to non-reproducible results.
	Training epochs on cpu:   0%|          | 0/100 [00:00<?, ?epoch/s]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training batches on cpu:  96%|█████████▌| 24/25 [00:00<00:00, 238.26batch/s]
	Training epochs on cpu:   1%|          | 1/100 [00:00<00:21,  4.64epoch/s, loss=1.38, prev_loss=nan]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:   2%|▏         | 2/100 [00:00<00:18,  5.19epoch/s, loss=1.33, prev_loss=1.38]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:   3%|▎         | 3/100 [00:00<00:18,  5.36epoch/s, loss=1.33, prev_loss=1.33]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:   4%|▍         | 4/100 [00:00<00:18,  5.26epoch/s, loss=1.29, prev_loss=1.33]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:   5%|▌         | 5/100 [00:00<00:16,  5.63epoch/s, loss=1.2, prev_loss=1.29]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:   6%|▌         | 6/100 [00:01<00:17,  5.36epoch/s, loss=1.17, prev_loss=1.2]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:   7%|▋         | 7/100 [00:01<00:16,  5.62epoch/s, loss=1.18, prev_loss=1.17]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:   8%|▊         | 8/100 [00:01<00:15,  5.77epoch/s, loss=1.1, prev_loss=1.18]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:   9%|▉         | 9/100 [00:01<00:16,  5.61epoch/s, loss=1.17, prev_loss=1.1]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  10%|█         | 10/100 [00:01<00:15,  5.83epoch/s, loss=1.11, prev_loss=1.17]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  11%|█         | 11/100 [00:01<00:14,  6.01epoch/s, loss=1.1, prev_loss=1.11]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training batches on cpu:  96%|█████████▌| 24/25 [00:00<00:00, 227.02batch/s]
	Training epochs on cpu:  12%|█▏        | 12/100 [00:02<00:16,  5.40epoch/s, loss=1.11, prev_loss=1.1]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  13%|█▎        | 13/100 [00:02<00:15,  5.63epoch/s, loss=1.01, prev_loss=1.11]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  14%|█▍        | 14/100 [00:02<00:14,  5.74epoch/s, loss=0.989, prev_loss=1.01]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  15%|█▌        | 15/100 [00:02<00:14,  5.83epoch/s, loss=1, prev_loss=0.989]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  16%|█▌        | 16/100 [00:02<00:14,  5.91epoch/s, loss=1.01, prev_loss=1]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  17%|█▋        | 17/100 [00:02<00:13,  6.20epoch/s, loss=0.978, prev_loss=1.01]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  18%|█▊        | 18/100 [00:03<00:13,  6.17epoch/s, loss=0.954, prev_loss=0.978]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  19%|█▉        | 19/100 [00:03<00:13,  6.19epoch/s, loss=0.917, prev_loss=0.954]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  20%|██        | 20/100 [00:03<00:12,  6.40epoch/s, loss=0.933, prev_loss=0.917]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  21%|██        | 21/100 [00:03<00:12,  6.11epoch/s, loss=0.945, prev_loss=0.933]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  22%|██▏       | 22/100 [00:03<00:13,  5.88epoch/s, loss=0.929, prev_loss=0.945]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  23%|██▎       | 23/100 [00:03<00:12,  6.18epoch/s, loss=0.933, prev_loss=0.929]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training batches on cpu: 100%|██████████| 25/25 [00:00<00:00, 243.37batch/s]
	Training epochs on cpu:  24%|██▍       | 24/100 [00:04<00:12,  5.94epoch/s, loss=0.885, prev_loss=0.933]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  25%|██▌       | 25/100 [00:04<00:12,  5.94epoch/s, loss=0.914, prev_loss=0.885]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  26%|██▌       | 26/100 [00:04<00:12,  6.11epoch/s, loss=0.892, prev_loss=0.914]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  27%|██▋       | 27/100 [00:04<00:12,  5.92epoch/s, loss=0.88, prev_loss=0.892]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  28%|██▊       | 28/100 [00:04<00:12,  5.81epoch/s, loss=0.887, prev_loss=0.88]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  29%|██▉       | 29/100 [00:04<00:11,  6.08epoch/s, loss=0.849, prev_loss=0.887]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  30%|███       | 30/100 [00:05<00:12,  5.78epoch/s, loss=0.829, prev_loss=0.849]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  31%|███       | 31/100 [00:05<00:12,  5.67epoch/s, loss=0.864, prev_loss=0.829]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  32%|███▏      | 32/100 [00:05<00:11,  5.77epoch/s, loss=0.845, prev_loss=0.864]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  33%|███▎      | 33/100 [00:05<00:11,  5.79epoch/s, loss=0.834, prev_loss=0.845]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  34%|███▍      | 34/100 [00:05<00:11,  5.85epoch/s, loss=0.83, prev_loss=0.834]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  35%|███▌      | 35/100 [00:06<00:11,  5.64epoch/s, loss=0.804, prev_loss=0.83]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  36%|███▌      | 36/100 [00:06<00:10,  5.92epoch/s, loss=0.803, prev_loss=0.804]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  37%|███▋      | 37/100 [00:06<00:10,  5.94epoch/s, loss=0.821, prev_loss=0.803]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training batches on cpu: 100%|██████████| 25/25 [00:00<00:00, 245.75batch/s]
	Training epochs on cpu:  38%|███▊      | 38/100 [00:06<00:11,  5.46epoch/s, loss=0.804, prev_loss=0.821]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  39%|███▉      | 39/100 [00:06<00:10,  5.72epoch/s, loss=0.797, prev_loss=0.804]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  40%|████      | 40/100 [00:06<00:10,  5.69epoch/s, loss=0.775, prev_loss=0.797]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  41%|████      | 41/100 [00:07<00:10,  5.69epoch/s, loss=0.767, prev_loss=0.775]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  42%|████▏     | 42/100 [00:07<00:09,  5.80epoch/s, loss=0.792, prev_loss=0.767]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  43%|████▎     | 43/100 [00:07<00:09,  5.71epoch/s, loss=0.776, prev_loss=0.792]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  44%|████▍     | 44/100 [00:07<00:09,  5.87epoch/s, loss=0.775, prev_loss=0.776]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  45%|████▌     | 45/100 [00:07<00:09,  5.88epoch/s, loss=0.79, prev_loss=0.775]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  46%|████▌     | 46/100 [00:07<00:09,  5.75epoch/s, loss=0.733, prev_loss=0.79]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  47%|████▋     | 47/100 [00:08<00:09,  5.72epoch/s, loss=0.775, prev_loss=0.733]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  48%|████▊     | 48/100 [00:08<00:09,  5.35epoch/s, loss=0.764, prev_loss=0.775]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  49%|████▉     | 49/100 [00:08<00:09,  5.20epoch/s, loss=0.79, prev_loss=0.764]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  50%|█████     | 50/100 [00:08<00:09,  5.35epoch/s, loss=0.746, prev_loss=0.79]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  51%|█████     | 51/100 [00:08<00:08,  5.61epoch/s, loss=0.733, prev_loss=0.746]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  52%|█████▏    | 52/100 [00:09<00:08,  5.60epoch/s, loss=0.736, prev_loss=0.733]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  53%|█████▎    | 53/100 [00:09<00:08,  5.75epoch/s, loss=0.719, prev_loss=0.736]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  54%|█████▍    | 54/100 [00:09<00:07,  5.79epoch/s, loss=0.739, prev_loss=0.719]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  55%|█████▌    | 55/100 [00:09<00:07,  6.08epoch/s, loss=0.726, prev_loss=0.739]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  56%|█████▌    | 56/100 [00:09<00:07,  6.05epoch/s, loss=0.731, prev_loss=0.726]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  57%|█████▋    | 57/100 [00:09<00:07,  5.69epoch/s, loss=0.743, prev_loss=0.731]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  58%|█████▊    | 58/100 [00:10<00:07,  5.80epoch/s, loss=0.726, prev_loss=0.743]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  59%|█████▉    | 59/100 [00:10<00:06,  5.91epoch/s, loss=0.759, prev_loss=0.726]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  60%|██████    | 60/100 [00:10<00:07,  5.68epoch/s, loss=0.736, prev_loss=0.759]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  61%|██████    | 61/100 [00:10<00:06,  5.87epoch/s, loss=0.733, prev_loss=0.736]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  62%|██████▏   | 62/100 [00:10<00:06,  5.53epoch/s, loss=0.709, prev_loss=0.733]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  63%|██████▎   | 63/100 [00:10<00:06,  5.59epoch/s, loss=0.716, prev_loss=0.709]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  64%|██████▍   | 64/100 [00:11<00:06,  5.78epoch/s, loss=0.722, prev_loss=0.716]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  65%|██████▌   | 65/100 [00:11<00:06,  5.69epoch/s, loss=0.737, prev_loss=0.722]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  66%|██████▌   | 66/100 [00:11<00:05,  5.97epoch/s, loss=0.684, prev_loss=0.737]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training batches on cpu: 100%|██████████| 25/25 [00:00<00:00, 243.37batch/s]
	Training epochs on cpu:  67%|██████▋   | 67/100 [00:11<00:05,  5.64epoch/s, loss=0.697, prev_loss=0.684]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  68%|██████▊   | 68/100 [00:11<00:05,  5.59epoch/s, loss=0.717, prev_loss=0.697]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  69%|██████▉   | 69/100 [00:12<00:05,  5.48epoch/s, loss=0.684, prev_loss=0.717]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  70%|███████   | 70/100 [00:12<00:05,  5.76epoch/s, loss=0.685, prev_loss=0.684]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  71%|███████   | 71/100 [00:12<00:05,  5.80epoch/s, loss=0.707, prev_loss=0.685]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  72%|███████▏  | 72/100 [00:12<00:04,  5.71epoch/s, loss=0.702, prev_loss=0.707]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  73%|███████▎  | 73/100 [00:12<00:04,  5.59epoch/s, loss=0.705, prev_loss=0.702]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  74%|███████▍  | 74/100 [00:12<00:04,  5.54epoch/s, loss=0.693, prev_loss=0.705]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  75%|███████▌  | 75/100 [00:13<00:04,  5.58epoch/s, loss=0.695, prev_loss=0.693]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  76%|███████▌  | 76/100 [00:13<00:04,  5.69epoch/s, loss=0.682, prev_loss=0.695]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  77%|███████▋  | 77/100 [00:13<00:04,  5.74epoch/s, loss=0.672, prev_loss=0.682]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  78%|███████▊  | 78/100 [00:13<00:03,  5.87epoch/s, loss=0.688, prev_loss=0.672]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  79%|███████▉  | 79/100 [00:13<00:03,  5.55epoch/s, loss=0.689, prev_loss=0.688]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  80%|████████  | 80/100 [00:13<00:03,  5.69epoch/s, loss=0.646, prev_loss=0.689]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training batches on cpu: 100%|██████████| 25/25 [00:00<00:00, 248.18batch/s]
	Training epochs on cpu:  81%|████████  | 81/100 [00:14<00:03,  5.44epoch/s, loss=0.673, prev_loss=0.646]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  82%|████████▏ | 82/100 [00:14<00:03,  5.39epoch/s, loss=0.671, prev_loss=0.673]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  83%|████████▎ | 83/100 [00:14<00:03,  5.35epoch/s, loss=0.66, prev_loss=0.671]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  84%|████████▍ | 84/100 [00:14<00:02,  5.39epoch/s, loss=0.66, prev_loss=0.66]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  85%|████████▌ | 85/100 [00:14<00:02,  5.60epoch/s, loss=0.665, prev_loss=0.66]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  86%|████████▌ | 86/100 [00:15<00:02,  5.61epoch/s, loss=0.679, prev_loss=0.665]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  87%|████████▋ | 87/100 [00:15<00:02,  5.38epoch/s, loss=0.658, prev_loss=0.679]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  88%|████████▊ | 88/100 [00:15<00:02,  5.68epoch/s, loss=0.692, prev_loss=0.658]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  89%|████████▉ | 89/100 [00:15<00:01,  5.68epoch/s, loss=0.675, prev_loss=0.692]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  90%|█████████ | 90/100 [00:15<00:01,  5.69epoch/s, loss=0.667, prev_loss=0.675]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  91%|█████████ | 91/100 [00:15<00:01,  5.77epoch/s, loss=0.674, prev_loss=0.667]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  92%|█████████▏| 92/100 [00:16<00:01,  5.93epoch/s, loss=0.689, prev_loss=0.674]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  93%|█████████▎| 93/100 [00:16<00:01,  5.53epoch/s, loss=0.659, prev_loss=0.689]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  94%|█████████▍| 94/100 [00:16<00:01,  5.55epoch/s, loss=0.674, prev_loss=0.659]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  95%|█████████▌| 95/100 [00:16<00:00,  5.62epoch/s, loss=0.654, prev_loss=0.674]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  96%|█████████▌| 96/100 [00:16<00:00,  5.50epoch/s, loss=0.641, prev_loss=0.654]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  97%|█████████▋| 97/100 [00:16<00:00,  5.83epoch/s, loss=0.668, prev_loss=0.641]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  98%|█████████▊| 98/100 [00:17<00:00,  5.83epoch/s, loss=0.643, prev_loss=0.668]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu:  99%|█████████▉| 99/100 [00:17<00:00,  5.98epoch/s, loss=0.651, prev_loss=0.643]
	Training batches on cpu:   0%|          | 0/25 [00:00<?, ?batch/s]
	Training epochs on cpu: 100%|██████████| 100/100 [00:17<00:00,  5.72epoch/s, loss=0.668, prev_loss=0.651]
	Evaluating on cpu:   0%|          | 0.00/201 [00:00<?, ?triple/s]Encountered tensors on device_types={'cpu'} while only ['cuda'] are considered safe for automatic memory utilization maximization. This may lead to undocumented crashes (but can be safe, too).
	Evaluating on cpu: 100%|██████████| 201/201 [00:00<00:00, 2.29ktriple/s]
	Предвидувања за "uk":

	Предвидени релации за "uk" со "accusation" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -7.597968           uk
	12       12  -9.099523          usa
	13       13  -9.169643         ussr
	1         1  -9.209929        burma
	7         7  -9.538837       israel
	2         2  -9.623510        china
	5         5  -9.661216        india
	9         9  -9.675089  netherlands
	3         3  -9.880312         cuba
	4         4 -10.024112        egypt
	0         0 -10.080850       brazil
	8         8 -10.298676       jordan
	10       10 -10.482346       poland
	6         6 -10.560797    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 0))

	Вистински релации за "uk" со "accusation" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "aidenemy" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	11       11 -6.685667           uk
	12       12 -6.895002          usa
	8         8 -7.639935       jordan
	9         9 -7.716454  netherlands
	5         5 -7.957656        india
	4         4 -7.995090        egypt
	0         0 -8.077215       brazil
	13       13 -8.096165         ussr
	1         1 -8.157153        burma
	7         7 -8.391644       israel
	6         6 -8.861476    indonesia
	3         3 -8.874524         cuba
	2         2 -9.033664        china
	10       10 -9.195134       poland, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 1))

	Вистински релации за "uk" со "aidenemy" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "attackembassy" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	11       11 -6.256288           uk
	12       12 -6.442486          usa
	9         9 -6.937387  netherlands
	4         4 -6.998789        egypt
	6         6 -7.182576    indonesia
	8         8 -7.294715       jordan
	7         7 -7.630892       israel
	0         0 -7.722954       brazil
	13       13 -7.752064         ussr
	2         2 -7.806118        china
	3         3 -8.369467         cuba
	10       10 -8.945576       poland
	5         5 -9.068976        india
	1         1 -9.261388        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 2))

	Вистински релации за "uk" со "attackembassy" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "blockpositionindex" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	10       10  -8.992908       poland
	11       11  -9.130253           uk
	5         5  -9.152133        india
	13       13  -9.200875         ussr
	12       12  -9.331532          usa
	2         2  -9.611148        china
	9         9  -9.780499  netherlands
	3         3  -9.781292         cuba
	7         7  -9.826075       israel
	4         4  -9.896627        egypt
	0         0 -10.355951       brazil
	6         6 -10.476033    indonesia
	1         1 -10.548476        burma
	8         8 -10.549150       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 3))

	Вистински релации за "uk" со "blockpositionindex" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "booktranslations" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.238850           uk
	13       13  -9.124201         ussr
	12       12  -9.173813          usa
	9         9  -9.998028  netherlands
	3         3 -10.244641         cuba
	2         2 -10.598183        china
	7         7 -10.721777       israel
	5         5 -10.836401        india
	10       10 -10.840928       poland
	0         0 -10.981953       brazil
	8         8 -11.288202       jordan
	6         6 -11.474212    indonesia
	4         4 -11.598377        egypt
	1         1 -12.417917        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 4))

	Вистински релации за "uk" со "booktranslations" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "boycottembargo" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	11       11 -6.343006           uk
	5         5 -7.423199        india
	4         4 -7.816699        egypt
	12       12 -7.878254          usa
	7         7 -8.064287       israel
	9         9 -8.256899  netherlands
	13       13 -8.436888         ussr
	10       10 -8.518076       poland
	0         0 -8.772740       brazil
	3         3 -8.965449         cuba
	8         8 -9.108926       jordan
	6         6 -9.176249    indonesia
	1         1 -9.324688        burma
	2         2 -9.861996        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 5))

	Вистински релации за "uk" со "boycottembargo" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "commonbloc0" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -7.954994           uk
	13       13  -8.181494         ussr
	10       10  -8.283414       poland
	12       12  -8.562392          usa
	2         2  -8.746344        china
	9         9  -9.060740  netherlands
	3         3  -9.081771         cuba
	5         5  -9.250044        india
	1         1  -9.278737        burma
	0         0  -9.403419       brazil
	7         7  -9.584876       israel
	4         4  -9.667258        egypt
	6         6  -9.792645    indonesia
	8         8 -10.437393       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 6))

	Вистински релации за "uk" со "commonbloc0" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "commonbloc1" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	4         4 -11.065439        egypt
	11       11 -11.318711           uk
	7         7 -11.409291       israel
	8         8 -11.411097       jordan
	2         2 -11.632789        china
	13       13 -11.664055         ussr
	5         5 -11.766613        india
	9         9 -11.852149  netherlands
	12       12 -11.907807          usa
	10       10 -11.913831       poland
	0         0 -11.930589       brazil
	6         6 -12.020270    indonesia
	1         1 -12.276561        burma
	3         3 -12.684951         cuba, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 7))

	Вистински релации за "uk" со "commonbloc1" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "commonbloc2" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.468760           uk
	9         9  -9.589799  netherlands
	0         0  -9.660507       brazil
	12       12  -9.755247          usa
	5         5 -10.012653        india
	3         3 -10.489809         cuba
	6         6 -10.503359    indonesia
	10       10 -10.547770       poland
	7         7 -10.567766       israel
	1         1 -10.629704        burma
	2         2 -10.688833        china
	13       13 -10.745856         ussr
	4         4 -11.429958        egypt
	8         8 -11.446834       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 8))

	Вистински релации за "uk" со "commonbloc2" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "conferences" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -9.495179           uk
	13       13  -9.800983         ussr
	12       12  -9.963963          usa
	5         5  -9.978717        india
	9         9 -10.412588  netherlands
	10       10 -10.473012       poland
	3         3 -10.540545         cuba
	1         1 -10.818047        burma
	0         0 -10.835982       brazil
	2         2 -10.878596        china
	7         7 -11.005378       israel
	4         4 -11.164657        egypt
	6         6 -11.619254    indonesia
	8         8 -11.732646       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 9))

	Вистински релации за "uk" со "conferences" релацијата:
	['indonesia']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "dependent" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -7.359798           uk
	13       13  -9.270459         ussr
	10       10  -9.528206       poland
	12       12  -9.557985          usa
	9         9  -9.576628  netherlands
	0         0  -9.994058       brazil
	7         7 -10.009352       israel
	5         5 -10.169939        india
	2         2 -10.673731        china
	3         3 -10.679579         cuba
	6         6 -11.038266    indonesia
	4         4 -11.274054        egypt
	1         1 -11.693810        burma
	8         8 -11.775982       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 10))

	Вистински релации за "uk" со "dependent" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "duration" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -7.534503           uk
	2         2  -8.524253        china
	10       10  -8.859422       poland
	13       13  -8.958028         ussr
	8         8  -9.063063       jordan
	5         5  -9.152832        india
	12       12  -9.156777          usa
	0         0  -9.237535       brazil
	4         4  -9.351465        egypt
	9         9  -9.380777  netherlands
	3         3  -9.428511         cuba
	1         1  -9.612785        burma
	7         7  -9.895109       israel
	6         6 -10.799813    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 11))

	Вистински релации за "uk" со "duration" релацијата:
	['indonesia']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "economicaid" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	7         7  -7.570824       israel
	8         8  -7.694344       jordan
	5         5  -7.928799        india
	0         0  -8.122783       brazil
	9         9  -8.251592  netherlands
	11       11  -8.328220           uk
	4         4  -8.488810        egypt
	12       12  -8.789154          usa
	6         6  -8.885599    indonesia
	3         3  -9.369045         cuba
	10       10  -9.414490       poland
	1         1  -9.712268        burma
	13       13 -10.170019         ussr
	2         2 -10.288900        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 12))

	Вистински релации за "uk" со "economicaid" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "eemigrants" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.762534          usa
	11       11  -6.764524           uk
	9         9  -7.791057  netherlands
	5         5  -8.176929        india
	0         0  -8.320977       brazil
	13       13  -8.404953         ussr
	4         4  -8.428617        egypt
	7         7  -8.605337       israel
	6         6  -8.885936    indonesia
	10       10  -8.983665       poland
	2         2  -9.383763        china
	3         3  -9.393812         cuba
	1         1  -9.764268        burma
	8         8 -10.136502       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 13))

	Вистински релации за "uk" со "eemigrants" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "embassy" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11 -10.361749           uk
	5         5 -10.410187        india
	12       12 -10.439652          usa
	7         7 -10.687730       israel
	2         2 -10.742332        china
	8         8 -10.826023       jordan
	9         9 -10.840173  netherlands
	0         0 -10.876397       brazil
	1         1 -10.893556        burma
	4         4 -10.986343        egypt
	13       13 -11.044807         ussr
	3         3 -11.464904         cuba
	6         6 -11.494727    indonesia
	10       10 -11.583140       poland, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 14))

	Вистински релации за "uk" со "embassy" релацијата:
	['netherlands']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "emigrants3" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.301073          usa
	11       11  -6.646591           uk
	5         5  -7.449763        india
	4         4  -7.629830        egypt
	13       13  -7.645758         ussr
	2         2  -8.049173        china
	9         9  -8.142373  netherlands
	0         0  -8.190423       brazil
	6         6  -8.349598    indonesia
	10       10  -8.721393       poland
	7         7  -8.804058       israel
	3         3  -9.140003         cuba
	8         8  -9.743828       jordan
	1         1 -10.160877        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 15))

	Вистински релации за "uk" со "emigrants3" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "expeldiplomats" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -6.870571           uk
	12       12  -7.112424          usa
	13       13  -7.146931         ussr
	9         9  -7.343054  netherlands
	7         7  -8.220803       israel
	2         2  -8.321285        china
	10       10  -8.347300       poland
	5         5  -8.365122        india
	3         3  -8.446698         cuba
	0         0  -8.761524       brazil
	6         6  -8.840861    indonesia
	4         4  -8.944074        egypt
	8         8  -9.650296       jordan
	1         1 -10.023687        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 16))

	Вистински релации за "uk" со "expeldiplomats" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "exportbooks" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	5         5 -6.730522        india
	11       11 -6.846952           uk
	0         0 -6.926783       brazil
	4         4 -7.209307        egypt
	12       12 -7.253837          usa
	1         1 -7.291535        burma
	2         2 -7.551254        china
	9         9 -7.972353  netherlands
	3         3 -8.217626         cuba
	10       10 -8.285079       poland
	13       13 -8.387346         ussr
	6         6 -8.563268    indonesia
	7         7 -8.703356       israel
	8         8 -9.033348       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 17))

	Вистински релации за "uk" со "exportbooks" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "exports3" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -7.779892           uk
	12       12  -8.833764          usa
	9         9  -8.843168  netherlands
	13       13  -8.918220         ussr
	5         5  -9.120091        india
	2         2  -9.637891        china
	10       10  -9.865936       poland
	7         7  -9.944860       israel
	4         4 -10.337238        egypt
	0         0 -10.650151       brazil
	3         3 -10.700820         cuba
	6         6 -10.954199    indonesia
	1         1 -12.047131        burma
	8         8 -12.516541       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 18))

	Вистински релации за "uk" со "exports3" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "independence" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -9.121414          usa
	3         3  -9.200286         cuba
	11       11  -9.254975           uk
	10       10  -9.296922       poland
	8         8  -9.563314       jordan
	0         0  -9.576889       brazil
	13       13  -9.637930         ussr
	4         4  -9.714290        egypt
	5         5  -9.757121        india
	9         9 -10.001373  netherlands
	2         2 -10.085427        china
	7         7 -10.124261       israel
	1         1 -10.348861        burma
	6         6 -10.653111    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 19))

	Вистински релации за "uk" со "independence" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "intergovorgs" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.967409           uk
	9         9  -9.439326  netherlands
	12       12  -9.446383          usa
	5         5  -9.504927        india
	10       10  -9.718107       poland
	0         0  -9.953006       brazil
	7         7 -10.255522       israel
	13       13 -10.529121         ussr
	2         2 -10.564939        china
	4         4 -10.732966        egypt
	3         3 -10.820068         cuba
	6         6 -11.040376    indonesia
	8         8 -11.481263       jordan
	1         1 -11.564960        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 20))

	Вистински релации за "uk" со "intergovorgs" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "intergovorgs3" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -9.298137           uk
	4         4  -9.948404        egypt
	12       12  -9.958825          usa
	5         5 -10.245848        india
	7         7 -10.472328       israel
	9         9 -10.538821  netherlands
	13       13 -10.833247         ussr
	10       10 -10.969449       poland
	0         0 -11.192683       brazil
	8         8 -11.598949       jordan
	3         3 -11.953491         cuba
	6         6 -11.989044    indonesia
	2         2 -12.079298        china
	1         1 -12.553572        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 21))

	Вистински релации за "uk" со "intergovorgs3" релацијата:
	['brazil', 'netherlands']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "lostterritory" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	11       11 -7.173805           uk
	8         8 -7.215905       jordan
	4         4 -7.278336        egypt
	13       13 -8.084661         ussr
	5         5 -8.108986        india
	12       12 -8.222770          usa
	7         7 -8.547182       israel
	1         1 -9.021993        burma
	2         2 -9.073831        china
	3         3 -9.134981         cuba
	9         9 -9.341119  netherlands
	0         0 -9.342850       brazil
	6         6 -9.354348    indonesia
	10       10 -9.577858       poland, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 22))

	Вистински релации за "uk" со "lostterritory" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "militaryactions" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -6.870127           uk
	12       12  -7.529879          usa
	6         6  -7.699583    indonesia
	9         9  -7.828331  netherlands
	13       13  -7.875240         ussr
	4         4  -8.100908        egypt
	2         2  -8.232101        china
	5         5  -8.450622        india
	8         8  -8.766846       jordan
	0         0  -9.036617       brazil
	10       10  -9.187881       poland
	7         7  -9.396949       israel
	1         1  -9.755668        burma
	3         3 -10.152063         cuba, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 23))

	Вистински релации за "uk" со "militaryactions" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "militaryalliance" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	11       11 -6.777037           uk
	12       12 -7.302862          usa
	5         5 -7.514284        india
	7         7 -7.834157       israel
	9         9 -7.987827  netherlands
	0         0 -8.206736       brazil
	4         4 -8.730880        egypt
	6         6 -8.787840    indonesia
	13       13 -8.916630         ussr
	10       10 -9.003308       poland
	3         3 -9.103299         cuba
	2         2 -9.234711        china
	8         8 -9.294077       jordan
	1         1 -9.555336        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 24))

	Вистински релации за "uk" со "militaryalliance" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "negativebehavior" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.287505           uk
	13       13  -9.309808         ussr
	5         5  -9.546569        india
	2         2  -9.597125        china
	12       12  -9.826285          usa
	10       10  -9.930312       poland
	9         9  -9.969963  netherlands
	7         7 -10.027227       israel
	4         4 -10.210713        egypt
	3         3 -10.353168         cuba
	6         6 -10.355472    indonesia
	0         0 -10.510573       brazil
	1         1 -10.601647        burma
	8         8 -10.952038       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 25))

	Вистински релации за "uk" со "negativebehavior" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "negativecomm" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	11       11 -7.347208           uk
	13       13 -7.539415         ussr
	3         3 -8.323061         cuba
	12       12 -8.547531          usa
	5         5 -8.721943        india
	10       10 -8.782099       poland
	0         0 -8.805344       brazil
	1         1 -8.819447        burma
	2         2 -8.855419        china
	6         6 -9.010364    indonesia
	8         8 -9.092707       jordan
	4         4 -9.296209        egypt
	7         7 -9.309147       israel
	9         9 -9.594136  netherlands, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 26))

	Вистински релации за "uk" со "negativecomm" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "ngo" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.504505           uk
	12       12  -8.612513          usa
	4         4  -8.795563        egypt
	13       13  -8.983262         ussr
	9         9  -9.031871  netherlands
	5         5  -9.042319        india
	0         0  -9.192448       brazil
	7         7  -9.636784       israel
	10       10  -9.758486       poland
	3         3  -9.850257         cuba
	2         2 -10.160529        china
	8         8 -10.413095       jordan
	1         1 -10.633235        burma
	6         6 -10.863714    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 27))

	Вистински релации за "uk" со "ngo" релацијата:
	['poland']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "ngoorgs3" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -9.888352           uk
	7         7  -9.901193       israel
	12       12  -9.996309          usa
	9         9 -10.138393  netherlands
	10       10 -10.222115       poland
	0         0 -10.394709       brazil
	5         5 -10.528447        india
	4         4 -11.067589        egypt
	3         3 -11.766654         cuba
	13       13 -11.810531         ussr
	8         8 -12.198062       jordan
	2         2 -12.237275        china
	1         1 -12.382341        burma
	6         6 -12.558735    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 28))

	Вистински релации за "uk" со "ngoorgs3" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "nonviolentbehavior" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	3         3  -7.467122         cuba
	11       11  -7.525654           uk
	12       12  -7.784617          usa
	13       13  -8.271428         ussr
	5         5  -8.667989        india
	7         7  -8.714282       israel
	4         4  -8.831780        egypt
	2         2  -8.904160        china
	0         0  -9.006757       brazil
	9         9  -9.034403  netherlands
	10       10  -9.384422       poland
	8         8  -9.467359       jordan
	1         1 -10.316835        burma
	6         6 -11.069284    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 29))

	Вистински релации за "uk" со "nonviolentbehavior" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "officialvisits" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	13       13  -8.093833         ussr
	11       11  -8.128462           uk
	10       10  -8.243142       poland
	12       12  -8.283527          usa
	5         5  -8.484437        india
	2         2  -8.957536        china
	4         4  -9.156140        egypt
	1         1  -9.333941        burma
	8         8  -9.358901       jordan
	9         9  -9.376462  netherlands
	7         7  -9.450624       israel
	0         0  -9.475869       brazil
	6         6 -10.139524    indonesia
	3         3 -10.150912         cuba, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 30))

	Вистински релации за "uk" со "officialvisits" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "pprotests" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	11       11 -7.746051           uk
	10       10 -7.906094       poland
	12       12 -7.997348          usa
	13       13 -8.078434         ussr
	4         4 -8.210473        egypt
	3         3 -8.371022         cuba
	9         9 -8.668995  netherlands
	7         7 -9.055106       israel
	5         5 -9.076062        india
	6         6 -9.163006    indonesia
	0         0 -9.172753       brazil
	1         1 -9.228779        burma
	2         2 -9.415615        china
	8         8 -9.632129       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 31))

	Вистински релации за "uk" со "pprotests" релацијата:
	['ussr']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "relbooktranslations" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.360429           uk
	12       12  -8.692309          usa
	13       13  -8.900895         ussr
	9         9  -9.427934  netherlands
	1         1  -9.961040        burma
	10       10 -10.363269       poland
	5         5 -10.418682        india
	4         4 -10.578283        egypt
	7         7 -10.854795       israel
	6         6 -10.861048    indonesia
	0         0 -10.933004       brazil
	3         3 -11.244824         cuba
	2         2 -11.701872        china
	8         8 -11.721079       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 32))

	Вистински релации за "uk" со "relbooktranslations" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "reldiplomacy" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11 -11.525480           uk
	9         9 -12.040069  netherlands
	5         5 -12.460322        india
	12       12 -12.614250          usa
	13       13 -13.104848         ussr
	7         7 -13.131957       israel
	4         4 -13.304636        egypt
	10       10 -13.354497       poland
	6         6 -13.362387    indonesia
	0         0 -13.486694       brazil
	1         1 -13.522144        burma
	3         3 -13.545270         cuba
	2         2 -13.603454        china
	8         8 -13.618260       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 33))

	Вистински релации за "uk" со "reldiplomacy" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "releconomicaid" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	8         8  -7.715158       jordan
	4         4  -7.875224        egypt
	5         5  -8.098574        india
	11       11  -8.290154           uk
	7         7  -8.531335       israel
	0         0  -9.079794       brazil
	1         1  -9.094224        burma
	9         9  -9.250631  netherlands
	10       10  -9.330861       poland
	12       12  -9.466242          usa
	6         6  -9.933168    indonesia
	13       13 -10.469171         ussr
	3         3 -10.852061         cuba
	2         2 -11.311330        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 34))

	Вистински релации за "uk" со "releconomicaid" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "relemigrants" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.182779          usa
	11       11  -7.030169           uk
	9         9  -7.372983  netherlands
	0         0  -7.778102       brazil
	5         5  -7.946227        india
	13       13  -7.975166         ussr
	2         2  -8.170225        china
	4         4  -8.283549        egypt
	1         1  -8.670830        burma
	6         6  -8.847566    indonesia
	10       10  -8.876010       poland
	7         7  -9.802170       israel
	8         8  -9.890155       jordan
	3         3 -10.302304         cuba, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 35))

	Вистински релации за "uk" со "relemigrants" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "relexportbooks" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	5         5 -6.309340        india
	9         9 -6.431730  netherlands
	12       12 -6.544684          usa
	11       11 -6.591882           uk
	1         1 -6.662510        burma
	0         0 -6.840303       brazil
	4         4 -7.142764        egypt
	7         7 -7.335069       israel
	2         2 -7.349772        china
	8         8 -7.790080       jordan
	13       13 -7.895488         ussr
	10       10 -8.109976       poland
	3         3 -8.201898         cuba
	6         6 -8.354618    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 36))

	Вистински релации за "uk" со "relexportbooks" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "relexports" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -7.508113           uk
	12       12  -8.138292          usa
	9         9  -8.375037  netherlands
	5         5  -8.751691        india
	13       13  -9.157110         ussr
	2         2  -9.257420        china
	4         4  -9.457207        egypt
	0         0  -9.564339       brazil
	7         7  -9.737775       israel
	10       10  -9.773232       poland
	3         3  -9.904348         cuba
	6         6 -10.116361    indonesia
	8         8 -11.474148       jordan
	1         1 -11.480299        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 37))

	Вистински релации за "uk" со "relexports" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "relintergovorgs" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11 -10.352342           uk
	9         9 -10.657476  netherlands
	7         7 -10.880072       israel
	12       12 -11.166414          usa
	5         5 -11.243314        india
	0         0 -11.573387       brazil
	10       10 -12.015140       poland
	4         4 -12.109031        egypt
	13       13 -12.701412         ussr
	8         8 -13.026137       jordan
	6         6 -13.184374    indonesia
	1         1 -13.191773        burma
	2         2 -13.207700        china
	3         3 -13.384986         cuba, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 38))

	Вистински релации за "uk" со "relintergovorgs" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "relngo" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11 -10.907235           uk
	9         9 -11.227820  netherlands
	7         7 -11.849971       israel
	12       12 -11.918875          usa
	0         0 -12.332018       brazil
	5         5 -12.350313        india
	10       10 -12.457635       poland
	13       13 -12.855514         ussr
	4         4 -13.183308        egypt
	3         3 -13.695359         cuba
	2         2 -13.801623        china
	1         1 -13.948682        burma
	6         6 -14.302681    indonesia
	8         8 -14.668321       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 39))

	Вистински релации за "uk" со "relngo" релацијата:
	['usa']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "relstudents" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.363721          usa
	11       11  -6.509625           uk
	13       13  -7.472160         ussr
	5         5  -8.023216        india
	9         9  -8.041407  netherlands
	7         7  -8.172937       israel
	0         0  -8.572016       brazil
	2         2  -8.816950        china
	10       10  -8.842211       poland
	3         3  -9.001656         cuba
	4         4  -9.270909        egypt
	1         1  -9.773043        burma
	6         6  -9.800063    indonesia
	8         8 -10.406327       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 40))

	Вистински релации за "uk" со "relstudents" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "reltourism" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	4         4 -7.377466        egypt
	7         7 -7.460739       israel
	11       11 -7.845973           uk
	5         5 -8.111979        india
	0         0 -8.219038       brazil
	12       12 -8.319711          usa
	10       10 -8.414115       poland
	8         8 -8.420203       jordan
	13       13 -8.501404         ussr
	9         9 -8.517992  netherlands
	1         1 -9.344920        burma
	6         6 -9.352354    indonesia
	2         2 -9.536916        china
	3         3 -9.967305         cuba, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 41))

	Вистински релации за "uk" со "reltourism" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "reltreaties" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -8.768641          usa
	11       11  -8.859084           uk
	9         9  -9.860746  netherlands
	5         5  -9.977846        india
	13       13 -10.021244         ussr
	10       10 -10.164578       poland
	7         7 -10.390202       israel
	2         2 -10.815437        china
	3         3 -10.913604         cuba
	0         0 -10.993451       brazil
	6         6 -11.117673    indonesia
	8         8 -11.580908       jordan
	1         1 -11.763647        burma
	4         4 -11.777339        egypt, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 42))

	Вистински релации за "uk" со "reltreaties" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "severdiplomatic" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -7.259769           uk
	9         9  -7.943133  netherlands
	12       12  -8.452222          usa
	3         3  -9.066377         cuba
	7         7  -9.115202       israel
	10       10  -9.386890       poland
	13       13  -9.448849         ussr
	0         0  -9.538125       brazil
	4         4  -9.899389        egypt
	2         2 -10.084598        china
	8         8 -10.117030       jordan
	5         5 -10.222007        india
	1         1 -10.517809        burma
	6         6 -11.275519    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 43))

	Вистински релации за "uk" со "severdiplomatic" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "students" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -7.050231           uk
	12       12  -7.427552          usa
	13       13  -8.094570         ussr
	9         9  -8.825254  netherlands
	5         5  -8.880793        india
	7         7  -9.097639       israel
	0         0  -9.199880       brazil
	2         2  -9.326360        china
	10       10  -9.506468       poland
	4         4  -9.964404        egypt
	3         3 -10.009658         cuba
	8         8 -10.564598       jordan
	6         6 -10.581809    indonesia
	1         1 -10.839023        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 44))

	Вистински релации за "uk" со "students" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "timesinceally" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -9.492819           uk
	13       13 -10.214716         ussr
	12       12 -10.264441          usa
	2         2 -10.381658        china
	10       10 -10.468195       poland
	7         7 -10.517103       israel
	5         5 -10.549697        india
	9         9 -10.589179  netherlands
	4         4 -10.652511        egypt
	3         3 -11.157819         cuba
	0         0 -11.329276       brazil
	1         1 -11.377194        burma
	6         6 -11.881940    indonesia
	8         8 -12.011161       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 45))

	Вистински релации за "uk" со "timesinceally" релацијата:
	['cuba', 'egypt', 'usa', 'ussr']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "timesincewar" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.559814           uk
	0         0  -8.807266       brazil
	2         2  -9.044145        china
	12       12  -9.209991          usa
	4         4  -9.235691        egypt
	5         5  -9.330306        india
	8         8  -9.397017       jordan
	7         7  -9.413949       israel
	9         9  -9.503504  netherlands
	13       13  -9.594164         ussr
	1         1  -9.670655        burma
	10       10 -10.054215       poland
	3         3 -10.071609         cuba
	6         6 -10.079702    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 46))

	Вистински релации за "uk" со "timesincewar" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "tourism" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	4         4 -6.989593        egypt
	11       11 -7.087704           uk
	12       12 -7.263918          usa
	7         7 -7.374818       israel
	5         5 -7.388965        india
	9         9 -7.634384  netherlands
	8         8 -7.675036       jordan
	13       13 -7.750091         ussr
	10       10 -8.400710       poland
	0         0 -8.491252       brazil
	1         1 -9.239531        burma
	3         3 -9.302650         cuba
	2         2 -9.362983        china
	6         6 -9.380175    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 47))

	Вистински релации за "uk" со "tourism" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "tourism3" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	11       11 -6.364041           uk
	12       12 -6.463930          usa
	9         9 -7.143515  netherlands
	5         5 -7.154526        india
	13       13 -7.195615         ussr
	7         7 -7.452191       israel
	10       10 -7.709649       poland
	0         0 -8.340665       brazil
	2         2 -8.391721        china
	4         4 -8.514928        egypt
	1         1 -8.663330        burma
	8         8 -8.723259       jordan
	3         3 -8.766946         cuba
	6         6 -9.461385    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 48))

	Вистински релации за "uk" со "tourism3" релацијата:
	['egypt']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "treaties" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.352109           uk
	12       12  -9.149569          usa
	9         9  -9.289362  netherlands
	7         7  -9.320243       israel
	5         5  -9.398139        india
	4         4  -9.540147        egypt
	13       13  -9.773242         ussr
	10       10  -9.848703       poland
	0         0 -10.047919       brazil
	3         3 -10.267431         cuba
	8         8 -10.274102       jordan
	6         6 -10.559620    indonesia
	1         1 -10.664287        burma
	2         2 -11.151656        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 49))

	Вистински релации за "uk" со "treaties" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "unoffialacts" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -6.929016           uk
	12       12  -7.491942          usa
	9         9  -8.532627  netherlands
	5         5  -8.770184        india
	0         0  -9.189807       brazil
	4         4  -9.228846        egypt
	13       13  -9.340085         ussr
	7         7  -9.342912       israel
	2         2  -9.500742        china
	3         3  -9.586322         cuba
	10       10  -9.706587       poland
	1         1  -9.861079        burma
	6         6 -10.078788    indonesia
	8         8 -10.346566       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 50))

	Вистински релации за "uk" со "unoffialacts" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "unweightedunvote" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.877792           uk
	7         7  -9.423398       israel
	13       13  -9.604364         ussr
	10       10  -9.846439       poland
	4         4  -9.855915        egypt
	5         5  -9.934072        india
	9         9 -10.027680  netherlands
	3         3 -10.212792         cuba
	0         0 -10.274611       brazil
	12       12 -10.524491          usa
	8         8 -10.616122       jordan
	6         6 -10.728919    indonesia
	1         1 -10.783119        burma
	2         2 -11.031077        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 51))

	Вистински релации за "uk" со "unweightedunvote" релацијата:
	['jordan']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "violentactions" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -6.993535           uk
	10       10  -7.951029       poland
	13       13  -8.487769         ussr
	9         9  -8.505898  netherlands
	2         2  -8.637602        china
	12       12  -8.699601          usa
	0         0  -8.707825       brazil
	5         5  -8.947074        india
	7         7  -9.004469       israel
	3         3  -9.083966         cuba
	4         4  -9.166933        egypt
	1         1  -9.917170        burma
	6         6 -10.998515    indonesia
	8         8 -11.247445       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 52))

	Вистински релации за "uk" со "violentactions" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "warning" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	6         6 -6.912206    indonesia
	11       11 -7.223970           uk
	4         4 -7.748346        egypt
	5         5 -7.752741        india
	12       12 -7.790288          usa
	8         8 -8.151747       jordan
	13       13 -8.407169         ussr
	1         1 -8.429198        burma
	9         9 -8.599747  netherlands
	7         7 -8.678974       israel
	0         0 -8.729109       brazil
	3         3 -9.327215         cuba
	10       10 -9.338001       poland
	2         2 -9.930731        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 53))

	Вистински релации за "uk" со "warning" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "uk" со "weightedunvote" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	5         5  -9.783638        india
	7         7  -9.976841       israel
	11       11 -10.200405           uk
	12       12 -10.589987          usa
	9         9 -10.599781  netherlands
	10       10 -10.726812       poland
	4         4 -10.769656        egypt
	13       13 -10.885322         ussr
	8         8 -10.902514       jordan
	1         1 -11.122237        burma
	3         3 -11.249104         cuba
	0         0 -11.449334       brazil
	2         2 -11.505521        china
	6         6 -11.776374    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(11, 54))

	Вистински релации за "uk" со "weightedunvote" релацијата:
	['cuba']


	────────────────────────────────────────────────────────────────────────────────────────────────────
	Предвидувања за "usa":

	Предвидени релации за "usa" со "accusation" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -7.597968          usa
	11       11  -8.600794           uk
	13       13  -8.606648         ussr
	2         2  -8.682177        china
	3         3  -9.102781         cuba
	9         9  -9.280705  netherlands
	5         5  -9.606549        india
	4         4  -9.641488        egypt
	6         6  -9.722819    indonesia
	1         1  -9.731213        burma
	0         0  -9.754981       brazil
	7         7 -10.002758       israel
	10       10 -10.236875       poland
	8         8 -10.300687       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 0))

	Вистински релации за "usa" со "accusation" релацијата:
	['cuba']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "aidenemy" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.685668          usa
	9         9  -8.133288  netherlands
	8         8  -8.405182       jordan
	5         5  -8.503946        india
	13       13  -8.758489         ussr
	0         0  -8.759572       brazil
	11       11  -8.767886           uk
	6         6  -8.787629    indonesia
	4         4  -8.840190        egypt
	7         7  -9.440295       israel
	1         1  -9.572008        burma
	2         2  -9.676367        china
	3         3  -9.733935         cuba
	10       10 -10.249904       poland, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 1))

	Вистински релации за "usa" со "aidenemy" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "attackembassy" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.256288          usa
	9         9  -7.345648  netherlands
	6         6  -7.476338    indonesia
	11       11  -8.048051           uk
	4         4  -8.128766        egypt
	8         8  -8.323804       jordan
	0         0  -8.380542       brazil
	2         2  -8.554029        china
	13       13  -8.723321         ussr
	7         7  -9.077730       israel
	3         3  -9.319358         cuba
	5         5  -9.446836        india
	10       10  -9.888691       poland
	1         1 -10.644556        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 2))

	Вистински релации за "usa" со "attackembassy" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "blockpositionindex" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -9.130252          usa
	5         5  -9.751156        india
	6         6  -9.866392    indonesia
	4         4  -9.923618        egypt
	13       13  -9.961218         ussr
	3         3 -10.117177         cuba
	2         2 -10.211103        china
	10       10 -10.368382       poland
	7         7 -10.459686       israel
	11       11 -10.592229           uk
	1         1 -10.709468        burma
	8         8 -10.729001       jordan
	9         9 -10.888949  netherlands
	0         0 -10.990525       brazil, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 3))

	Вистински релации за "usa" со "blockpositionindex" релацијата:
	['ussr']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "booktranslations" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.192323           uk
	12       12  -8.238850          usa
	13       13  -8.613737         ussr
	3         3  -9.221411         cuba
	9         9  -9.600981  netherlands
	2         2  -9.867974        china
	10       10 -10.074459       poland
	0         0 -10.121263       brazil
	5         5 -10.128738        india
	6         6 -10.430979    indonesia
	7         7 -10.436188       israel
	8         8 -10.779900       jordan
	4         4 -10.882138        egypt
	1         1 -11.618321        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 4))

	Вистински релации за "usa" со "booktranslations" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "boycottembargo" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	12       12 -6.343006          usa
	4         4 -7.219062        egypt
	5         5 -7.513110        india
	11       11 -7.610318           uk
	13       13 -7.843950         ussr
	9         9 -8.046401  netherlands
	6         6 -8.106827    indonesia
	7         7 -8.696841       israel
	3         3 -8.766341         cuba
	10       10 -8.860959       poland
	8         8 -8.863521       jordan
	0         0 -8.918986       brazil
	1         1 -9.650017        burma
	2         2 -9.710892        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 5))

	Вистински релации за "usa" со "boycottembargo" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "commonbloc0" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -7.954994          usa
	11       11  -8.396623           uk
	10       10  -8.600129       poland
	13       13  -8.602382         ussr
	2         2  -8.900738        china
	5         5  -9.108738        india
	9         9  -9.202436  netherlands
	3         3  -9.266784         cuba
	0         0  -9.554820       brazil
	6         6  -9.879937    indonesia
	1         1 -10.007346        burma
	4         4 -10.098115        egypt
	7         7 -10.197271       israel
	8         8 -10.536339       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 6))

	Вистински релации за "usa" со "commonbloc0" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "commonbloc1" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	4         4 -10.721470        egypt
	8         8 -11.298327       jordan
	12       12 -11.318711          usa
	2         2 -11.354507        china
	6         6 -11.506455    indonesia
	7         7 -11.640744       israel
	5         5 -11.659401        india
	13       13 -11.713423         ussr
	0         0 -11.747121       brazil
	11       11 -11.778287           uk
	10       10 -11.811390       poland
	9         9 -11.982474  netherlands
	3         3 -12.048832         cuba
	1         1 -12.684237        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 7))

	Вистински релации за "usa" со "commonbloc1" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "commonbloc2" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.451807           uk
	12       12  -8.468760          usa
	0         0  -8.757642       brazil
	9         9  -8.783809  netherlands
	13       13  -9.309368         ussr
	10       10  -9.324937       poland
	5         5  -9.358193        india
	6         6  -9.453391    indonesia
	3         3  -9.460776         cuba
	7         7  -9.560211       israel
	2         2  -9.599891        china
	4         4 -10.335469        egypt
	1         1 -10.562173        burma
	8         8 -10.749294       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 8))

	Вистински релации за "usa" со "commonbloc2" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "conferences" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -9.487127           uk
	12       12  -9.495179          usa
	13       13  -9.788330         ussr
	5         5  -9.868147        india
	9         9 -10.235679  netherlands
	3         3 -10.382312         cuba
	10       10 -10.412703       poland
	0         0 -10.442595       brazil
	1         1 -10.936695        burma
	4         4 -10.973463        egypt
	2         2 -11.092626        china
	7         7 -11.220602       israel
	8         8 -11.652051       jordan
	6         6 -11.675690    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 9))

	Вистински релации за "usa" со "conferences" релацијата:
	['indonesia']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "dependent" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -7.281445           uk
	12       12  -7.359797          usa
	9         9  -7.851965  netherlands
	13       13  -7.893652         ussr
	10       10  -8.082185       poland
	5         5  -8.820001        india
	0         0  -8.831278       brazil
	3         3  -8.963765         cuba
	2         2  -9.181980        china
	7         7  -9.432898       israel
	6         6  -9.441693    indonesia
	4         4  -9.932393        egypt
	1         1 -10.827368        burma
	8         8 -10.915694       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 10))

	Вистински релации за "usa" со "dependent" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "duration" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	2         2  -7.291364        china
	12       12  -7.534504          usa
	10       10  -7.979260       poland
	13       13  -8.279448         ussr
	11       11  -8.390600           uk
	5         5  -8.419465        india
	9         9  -8.565187  netherlands
	3         3  -8.609376         cuba
	0         0  -8.610101       brazil
	8         8  -8.840986       jordan
	4         4  -8.904507        egypt
	6         6  -9.201505    indonesia
	7         7  -9.586008       israel
	1         1 -10.009764        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 11))

	Вистински релации за "usa" со "duration" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "economicaid" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	0         0 -7.128178       brazil
	7         7 -7.399323       israel
	8         8 -7.650188       jordan
	5         5 -7.683133        india
	4         4 -7.840254        egypt
	12       12 -8.328220          usa
	9         9 -8.374176  netherlands
	6         6 -8.530277    indonesia
	3         3 -8.714046         cuba
	11       11 -8.764082           uk
	10       10 -9.025382       poland
	13       13 -9.745249         ussr
	1         1 -9.920321        burma
	2         2 -9.966265        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 12))

	Вистински релации за "usa" со "economicaid" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "eemigrants" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.764524          usa
	11       11  -7.525987           uk
	9         9  -7.916479  netherlands
	0         0  -8.336863       brazil
	5         5  -8.392371        india
	4         4  -8.825473        egypt
	13       13  -8.872270         ussr
	6         6  -8.903621    indonesia
	7         7  -9.173399       israel
	3         3  -9.246363         cuba
	10       10  -9.418802       poland
	2         2  -9.754232        china
	1         1  -9.797132        burma
	8         8 -10.292290       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 13))

	Вистински релации за "usa" со "eemigrants" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "embassy" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	5         5 -10.241978        india
	12       12 -10.361749          usa
	9         9 -10.480074  netherlands
	8         8 -10.650414       jordan
	2         2 -10.828724        china
	0         0 -10.837543       brazil
	11       11 -10.851628           uk
	13       13 -10.888721         ussr
	4         4 -10.901781        egypt
	1         1 -10.947429        burma
	7         7 -11.003838       israel
	6         6 -11.430449    indonesia
	10       10 -11.537102       poland
	3         3 -11.538215         cuba, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 14))

	Вистински релации за "usa" со "embassy" релацијата:
	['brazil', 'uk']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "emigrants3" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.646591          usa
	11       11  -7.583301           uk
	13       13  -8.597563         ussr
	0         0  -8.718773       brazil
	5         5  -8.822959        india
	9         9  -8.833686  netherlands
	4         4  -8.924506        egypt
	6         6  -9.012740    indonesia
	2         2  -9.121750        china
	3         3  -9.272247         cuba
	10       10  -9.576870       poland
	7         7  -9.634642       israel
	8         8 -10.718637       jordan
	1         1 -10.803839        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 15))

	Вистински релации за "usa" со "emigrants3" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "expeldiplomats" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.870571          usa
	9         9  -7.866073  netherlands
	13       13  -8.069788         ussr
	11       11  -8.235861           uk
	5         5  -8.770127        india
	6         6  -8.820073    indonesia
	0         0  -8.839428       brazil
	2         2  -8.843701        china
	3         3  -8.893220         cuba
	10       10  -9.121042       poland
	7         7  -9.323087       israel
	4         4  -9.391518        egypt
	8         8 -10.015508       jordan
	1         1 -10.829514        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 16))

	Вистински релации за "usa" со "expeldiplomats" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "exportbooks" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	5         5 -6.477481        india
	12       12 -6.846953          usa
	0         0 -7.007999       brazil
	9         9 -7.239351  netherlands
	11       11 -7.457831           uk
	2         2 -7.483295        china
	3         3 -7.744822         cuba
	4         4 -7.904253        egypt
	1         1 -8.007212        burma
	10       10 -8.172821       poland
	6         6 -8.357816    indonesia
	13       13 -8.395588         ussr
	8         8 -9.211632       jordan
	7         7 -9.234919       israel, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 17))

	Вистински релации за "usa" со "exportbooks" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "exports3" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -7.779893          usa
	13       13  -7.957849         ussr
	11       11  -8.364119           uk
	9         9  -8.488381  netherlands
	5         5  -8.792500        india
	2         2  -9.289257        china
	4         4  -9.732920        egypt
	10       10  -9.769081       poland
	3         3  -9.994984         cuba
	7         7 -10.002476       israel
	0         0 -10.372873       brazil
	6         6 -10.560630    indonesia
	8         8 -11.631881       jordan
	1         1 -11.828127        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 18))

	Вистински релации за "usa" со "exports3" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "independence" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -9.254975          usa
	3         3  -9.558810         cuba
	0         0  -9.614918       brazil
	13       13  -9.725109         ussr
	8         8  -9.782435       jordan
	4         4  -9.834379        egypt
	11       11  -9.885234           uk
	9         9  -9.897448  netherlands
	10       10 -10.004541       poland
	5         5 -10.011224        india
	1         1 -10.659806        burma
	7         7 -10.728022       israel
	6         6 -10.744494    indonesia
	2         2 -10.842206        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 19))

	Вистински релации за "usa" со "independence" релацијата:
	['china']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "intergovorgs" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.642774           uk
	9         9  -8.887556  netherlands
	12       12  -8.967409          usa
	5         5  -9.298235        india
	10       10  -9.346869       poland
	0         0  -9.402628       brazil
	7         7  -9.766641       israel
	13       13 -10.062450         ussr
	6         6 -10.269320    indonesia
	3         3 -10.320353         cuba
	2         2 -10.365814        china
	4         4 -10.438324        egypt
	1         1 -10.982383        burma
	8         8 -11.141180       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 20))

	Вистински релации за "usa" со "intergovorgs" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "intergovorgs3" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -9.199278           uk
	12       12  -9.298137          usa
	5         5  -9.808551        india
	9         9  -9.823121  netherlands
	4         4  -9.832825        egypt
	7         7  -9.952023       israel
	0         0 -10.365768       brazil
	13       13 -10.385485         ussr
	10       10 -10.842880       poland
	3         3 -11.002430         cuba
	8         8 -11.200732       jordan
	6         6 -11.252923    indonesia
	2         2 -11.553917        china
	1         1 -12.296302        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 21))

	Вистински релации за "usa" со "intergovorgs3" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "lostterritory" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	12       12 -7.173806          usa
	8         8 -7.185876       jordan
	5         5 -7.529244        india
	4         4 -7.625566        egypt
	13       13 -7.848168         ussr
	11       11 -8.154669           uk
	3         3 -8.665203         cuba
	6         6 -8.716968    indonesia
	7         7 -8.758235       israel
	2         2 -8.836137        china
	9         9 -9.084458  netherlands
	1         1 -9.228644        burma
	0         0 -9.385632       brazil
	10       10 -9.567766       poland, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 22))

	Вистински релации за "usa" со "lostterritory" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "militaryactions" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.870127          usa
	2         2  -7.316514        china
	6         6  -8.245126    indonesia
	9         9  -8.332577  netherlands
	11       11  -8.371582           uk
	5         5  -8.420456        india
	13       13  -8.499587         ussr
	4         4  -8.528228        egypt
	0         0  -9.003613       brazil
	8         8  -9.173763       jordan
	10       10  -9.414150       poland
	3         3 -10.052485         cuba
	7         7 -10.303481       israel
	1         1 -10.516307        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 23))

	Вистински релации за "usa" со "militaryactions" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "militaryalliance" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	12       12 -6.777037          usa
	5         5 -7.297014        india
	11       11 -7.348152           uk
	0         0 -7.511987       brazil
	6         6 -7.683822    indonesia
	7         7 -7.757852       israel
	9         9 -8.276677  netherlands
	3         3 -8.338567         cuba
	4         4 -8.520732        egypt
	13       13 -8.583680         ussr
	10       10 -8.777391       poland
	1         1 -8.930403        burma
	2         2 -8.999535        china
	8         8 -9.019272       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 24))

	Вистински релации за "usa" со "militaryalliance" релацијата:
	['netherlands', 'uk']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "negativebehavior" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -8.287505          usa
	2         2  -8.331941        china
	13       13  -8.509266         ussr
	11       11  -8.838962           uk
	3         3  -8.934071         cuba
	5         5  -9.015292        india
	9         9  -9.189843  netherlands
	10       10  -9.426174       poland
	6         6  -9.827991    indonesia
	0         0  -9.880465       brazil
	4         4 -10.124163        egypt
	1         1 -10.418163        burma
	7         7 -10.435666       israel
	8         8 -10.712927       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 25))

	Вистински релации за "usa" со "negativebehavior" релацијата:
	['egypt']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "negativecomm" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	13       13 -7.339657         ussr
	12       12 -7.347208          usa
	3         3 -8.200605         cuba
	6         6 -8.476383    indonesia
	2         2 -8.575393        china
	5         5 -8.676456        india
	4         4 -8.707019        egypt
	11       11 -8.796711           uk
	10       10 -8.989761       poland
	0         0 -9.206749       brazil
	8         8 -9.347531       jordan
	9         9 -9.611062  netherlands
	1         1 -9.782305        burma
	7         7 -9.865117       israel, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 26))

	Вистински релации за "usa" со "negativecomm" релацијата:
	['brazil']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "ngo" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -8.504505          usa
	9         9  -8.778439  netherlands
	13       13  -8.913177         ussr
	11       11  -9.138568           uk
	0         0  -9.494981       brazil
	5         5  -9.505705        india
	3         3  -9.520862         cuba
	4         4  -9.664018        egypt
	7         7  -9.675368       israel
	10       10  -9.735848       poland
	2         2 -10.402267        china
	6         6 -10.612412    indonesia
	8         8 -10.704979       jordan
	1         1 -11.034015        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 27))

	Вистински релации за "usa" со "ngo" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "ngoorgs3" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -9.888352          usa
	7         7 -10.170522       israel
	0         0 -10.400126       brazil
	9         9 -10.406643  netherlands
	11       11 -10.471756           uk
	5         5 -10.633302        india
	10       10 -10.788459       poland
	4         4 -11.495087        egypt
	3         3 -11.585621         cuba
	6         6 -12.041033    indonesia
	13       13 -12.088139         ussr
	8         8 -12.197691       jordan
	2         2 -12.207066        china
	1         1 -12.464628        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 28))

	Вистински релации за "usa" со "ngoorgs3" релацијата:
	['netherlands', 'uk']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "nonviolentbehavior" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	3         3  -7.191188         cuba
	12       12  -7.525654          usa
	13       13  -8.408488         ussr
	11       11  -8.564099           uk
	5         5  -8.603950        india
	9         9  -8.638317  netherlands
	2         2  -9.043847        china
	0         0  -9.072163       brazil
	10       10  -9.417807       poland
	4         4  -9.609549        egypt
	7         7  -9.728181       israel
	8         8 -10.176653       jordan
	6         6 -10.721006    indonesia
	1         1 -10.865477        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 29))

	Вистински релации за "usa" со "nonviolentbehavior" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "officialvisits" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -8.128462          usa
	5         5  -8.648143        india
	13       13  -8.900208         ussr
	10       10  -8.932522       poland
	4         4  -9.078096        egypt
	0         0  -9.247242       brazil
	11       11  -9.486256           uk
	9         9  -9.615485  netherlands
	2         2  -9.754504        china
	3         3  -9.909905         cuba
	1         1 -10.038934        burma
	6         6 -10.064794    indonesia
	7         7 -10.216063       israel
	8         8 -10.246325       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 30))

	Вистински релации за "usa" со "officialvisits" релацијата:
	['indonesia']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "pprotests" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -7.746051          usa
	13       13  -8.519307         ussr
	6         6  -8.743345    indonesia
	9         9  -8.906790  netherlands
	3         3  -9.058350         cuba
	4         4  -9.154395        egypt
	10       10  -9.169152       poland
	11       11  -9.367063           uk
	0         0  -9.499067       brazil
	5         5  -9.714890        india
	2         2  -9.818047        china
	1         1  -9.911175        burma
	7         7 -10.314981       israel
	8         8 -10.371607       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 31))

	Вистински релации за "usa" со "pprotests" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "relbooktranslations" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -8.360429          usa
	11       11  -9.085212           uk
	9         9  -9.407731  netherlands
	13       13  -9.458184         ussr
	5         5 -10.114450        india
	6         6 -10.348359    indonesia
	1         1 -10.418707        burma
	10       10 -10.998081       poland
	3         3 -11.133661         cuba
	0         0 -11.207317       brazil
	4         4 -11.219650        egypt
	2         2 -11.650806        china
	7         7 -11.725118       israel
	8         8 -12.145037       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 32))

	Вистински релации за "usa" со "relbooktranslations" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "reldiplomacy" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11 -11.293513           uk
	9         9 -11.456346  netherlands
	12       12 -11.525481          usa
	5         5 -11.811884        india
	13       13 -12.055850         ussr
	10       10 -12.347049       poland
	3         3 -12.444571         cuba
	7         7 -12.546539       israel
	4         4 -12.695945        egypt
	6         6 -12.753691    indonesia
	1         1 -12.790894        burma
	2         2 -12.914885        china
	8         8 -12.928214       jordan
	0         0 -13.085058       brazil, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 33))

	Вистински релации за "usa" со "reldiplomacy" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "releconomicaid" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	4         4  -6.890861        egypt
	8         8  -7.170808       jordan
	5         5  -7.678179        india
	7         7  -7.806947       israel
	0         0  -7.838555       brazil
	12       12  -8.290154          usa
	11       11  -8.686023           uk
	9         9  -8.694936  netherlands
	6         6  -8.784793    indonesia
	1         1  -8.890816        burma
	10       10  -8.933585       poland
	13       13  -9.892450         ussr
	3         3 -10.085707         cuba
	2         2 -10.247721        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 34))

	Вистински релации за "usa" со "releconomicaid" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "relemigrants" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -7.030169          usa
	9         9  -8.587474  netherlands
	0         0  -8.673044       brazil
	11       11  -9.025888           uk
	5         5  -9.143377        india
	13       13  -9.383024         ussr
	2         2  -9.441353        china
	4         4  -9.520000        egypt
	6         6  -9.677158    indonesia
	10       10 -10.192542       poland
	1         1 -10.492448        burma
	3         3 -10.674170         cuba
	7         7 -11.109467       israel
	8         8 -11.374520       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 35))

	Вистински релации за "usa" со "relemigrants" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "relexportbooks" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	12       12 -6.591881          usa
	5         5 -6.760745        india
	0         0 -6.936849       brazil
	9         9 -6.997568  netherlands
	1         1 -7.816990        burma
	11       11 -7.899673           uk
	4         4 -7.929010        egypt
	3         3 -8.098224         cuba
	10       10 -8.160995       poland
	2         2 -8.180289        china
	13       13 -8.210042         ussr
	7         7 -8.293510       israel
	6         6 -8.716582    indonesia
	8         8 -9.204653       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 36))

	Вистински релации за "usa" со "relexportbooks" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "relexports" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -7.508113          usa
	9         9  -8.129962  netherlands
	5         5  -8.239816        india
	11       11  -8.262692           uk
	13       13  -8.882511         ussr
	4         4  -9.269761        egypt
	10       10  -9.429641       poland
	2         2  -9.433453        china
	0         0  -9.605028       brazil
	7         7  -9.609052       israel
	3         3  -9.654679         cuba
	6         6 -10.008836    indonesia
	8         8 -11.436440       jordan
	1         1 -11.613487        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 37))

	Вистински релации за "usa" со "relexports" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "relintergovorgs" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11 -10.201169           uk
	9         9 -10.316060  netherlands
	12       12 -10.352342          usa
	7         7 -10.416832       israel
	5         5 -10.800654        india
	0         0 -10.806334       brazil
	4         4 -11.049658        egypt
	10       10 -11.345995       poland
	13       13 -11.872300         ussr
	1         1 -12.370840        burma
	8         8 -12.421219       jordan
	6         6 -12.458322    indonesia
	2         2 -12.478013        china
	3         3 -12.713554         cuba, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 38))

	Вистински релации за "usa" со "relintergovorgs" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "relngo" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11 -10.234384           uk
	9         9 -10.373418  netherlands
	12       12 -10.907235          usa
	7         7 -10.942688       israel
	0         0 -11.246458       brazil
	5         5 -11.323092        india
	10       10 -11.868887       poland
	13       13 -12.135829         ussr
	4         4 -12.228500        egypt
	3         3 -12.989927         cuba
	2         2 -12.998874        china
	6         6 -13.265409    indonesia
	1         1 -13.283421        burma
	8         8 -13.667338       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 39))

	Вистински релации за "usa" со "relngo" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "relstudents" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.509624          usa
	11       11  -7.674859           uk
	13       13  -8.152781         ussr
	5         5  -8.616413        india
	9         9  -8.616737  netherlands
	0         0  -8.903749       brazil
	3         3  -9.068184         cuba
	7         7  -9.378279       israel
	2         2  -9.565450        china
	6         6  -9.687490    indonesia
	10       10  -9.718381       poland
	4         4  -9.719852        egypt
	1         1 -10.744343        burma
	8         8 -11.181223       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 40))

	Вистински релации за "usa" со "relstudents" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "reltourism" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	7         7  -7.241798       israel
	12       12  -7.845973          usa
	4         4  -7.953673        egypt
	5         5  -7.974708        india
	11       11  -8.064499           uk
	8         8  -8.273065       jordan
	9         9  -8.279778  netherlands
	13       13  -8.345940         ussr
	0         0  -8.747458       brazil
	10       10  -8.877941       poland
	6         6  -9.175406    indonesia
	2         2  -9.297016        china
	3         3 -10.051641         cuba
	1         1 -10.094140        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 41))

	Вистински релации за "usa" со "reltourism" релацијата:
	['egypt', 'india']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "reltreaties" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -8.859085          usa
	11       11  -9.481259           uk
	5         5 -10.526205        india
	6         6 -10.557201    indonesia
	9         9 -10.587443  netherlands
	13       13 -10.923615         ussr
	7         7 -11.003157       israel
	3         3 -11.026687         cuba
	0         0 -11.091110       brazil
	10       10 -11.103064       poland
	2         2 -11.236527        china
	4         4 -11.759751        egypt
	1         1 -11.815853        burma
	8         8 -12.093922       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 42))

	Вистински релации за "usa" со "reltreaties" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "severdiplomatic" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -7.259769          usa
	9         9  -7.717676  netherlands
	11       11  -7.942485           uk
	3         3  -8.342301         cuba
	7         7  -8.591403       israel
	0         0  -8.828700       brazil
	13       13  -8.908354         ussr
	2         2  -9.087976        china
	10       10  -9.240927       poland
	5         5  -9.328481        india
	4         4  -9.624987        egypt
	8         8  -9.656396       jordan
	1         1 -10.129477        burma
	6         6 -10.474300    indonesia, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 43))

	Вистински релации за "usa" со "severdiplomatic" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "students" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -7.050231          usa
	11       11  -7.621718           uk
	13       13  -8.144378         ussr
	9         9  -8.925736  netherlands
	2         2  -9.149084        china
	5         5  -9.163879        india
	0         0  -9.371759       brazil
	3         3  -9.424967         cuba
	4         4  -9.435778        egypt
	7         7  -9.532950       israel
	10       10  -9.627123       poland
	6         6  -9.866818    indonesia
	8         8 -10.443159       jordan
	1         1 -10.820045        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 44))

	Вистински релации за "usa" со "students" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "timesinceally" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -9.492819          usa
	13       13  -9.898550         ussr
	11       11 -10.004560           uk
	5         5 -10.081651        india
	4         4 -10.442647        egypt
	2         2 -10.523464        china
	10       10 -10.610791       poland
	9         9 -10.786866  netherlands
	7         7 -10.973006       israel
	3         3 -11.271637         cuba
	6         6 -11.483154    indonesia
	0         0 -11.529902       brazil
	1         1 -11.535280        burma
	8         8 -11.754177       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 45))

	Вистински релации за "usa" со "timesinceally" релацијата:
	['cuba', 'netherlands']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "timesincewar" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -8.559814          usa
	0         0  -8.570269       brazil
	2         2  -8.734875        china
	9         9  -9.145686  netherlands
	3         3  -9.154323         cuba
	5         5  -9.476480        india
	6         6  -9.564031    indonesia
	4         4  -9.629878        egypt
	13       13  -9.682036         ussr
	8         8  -9.755006       jordan
	7         7  -9.784908       israel
	11       11  -9.882490           uk
	10       10 -10.262155       poland
	1         1 -10.628953        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 46))

	Вистински релации за "usa" со "timesincewar" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "tourism" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -7.087704          usa
	5         5  -7.544611        india
	4         4  -7.623307        egypt
	9         9  -7.875611  netherlands
	0         0  -8.130920       brazil
	7         7  -8.159098       israel
	11       11  -8.221645           uk
	13       13  -8.534842         ussr
	8         8  -8.634056       jordan
	6         6  -9.284822    indonesia
	10       10  -9.298129       poland
	3         3  -9.491202         cuba
	2         2  -9.764387        china
	1         1 -10.026621        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 47))

	Вистински релации за "usa" со "tourism" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "tourism3" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	12       12 -6.364041          usa
	5         5 -7.253855        india
	9         9 -7.586820  netherlands
	11       11 -7.635082           uk
	13       13 -7.784441         ussr
	7         7 -8.196320       israel
	10       10 -8.200995       poland
	0         0 -8.421171       brazil
	2         2 -8.636920        china
	3         3 -8.707150         cuba
	4         4 -8.776331        egypt
	6         6 -9.109266    indonesia
	1         1 -9.146073        burma
	8         8 -9.776092       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 48))

	Вистински релации за "usa" со "tourism3" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "treaties" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.348378           uk
	12       12  -8.352109          usa
	9         9  -8.408813  netherlands
	5         5  -8.612906        india
	4         4  -8.956745        egypt
	6         6  -9.122271    indonesia
	13       13  -9.135330         ussr
	3         3  -9.200537         cuba
	7         7  -9.349316       israel
	10       10  -9.357790       poland
	0         0  -9.421148       brazil
	8         8 -10.010434       jordan
	2         2 -10.251364        china
	1         1 -10.511117        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 49))

	Вистински релации за "usa" со "treaties" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "unoffialacts" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.929016          usa
	11       11  -8.386375           uk
	9         9  -8.544961  netherlands
	2         2  -9.078800        china
	3         3  -9.167918         cuba
	5         5  -9.169581        india
	6         6  -9.242509    indonesia
	0         0  -9.271525       brazil
	13       13  -9.465314         ussr
	10       10  -9.820053       poland
	4         4  -9.863341        egypt
	7         7 -10.200253       israel
	1         1 -10.383039        burma
	8         8 -11.010349       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 50))

	Вистински релации за "usa" со "unoffialacts" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "unweightedunvote" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	11       11  -8.659870           uk
	13       13  -8.668323         ussr
	12       12  -8.877792          usa
	4         4  -9.206520        egypt
	7         7  -9.261475       israel
	5         5  -9.409572        india
	0         0  -9.445068       brazil
	6         6  -9.559546    indonesia
	10       10  -9.669656       poland
	9         9  -9.706864  netherlands
	3         3  -9.772447         cuba
	8         8  -9.822602       jordan
	2         2  -9.994437        china
	1         1 -10.433484        burma, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 51))

	Вистински релации за "usa" со "unweightedunvote" релацијата:
	['israel']


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "violentactions" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	12       12  -6.993535          usa
	2         2  -7.345644        china
	10       10  -7.848650       poland
	9         9  -7.857568  netherlands
	5         5  -8.011315        india
	0         0  -8.170765       brazil
	13       13  -8.233674         ussr
	3         3  -8.239828         cuba
	11       11  -8.426276           uk
	4         4  -8.889632        egypt
	7         7  -9.139793       israel
	6         6  -9.945139    indonesia
	1         1 -10.213169        burma
	8         8 -11.019438       jordan, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 52))

	Вистински релации за "usa" со "violentactions" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "warning" релацијата

	TargetPredictions(df=    tail_id     score   tail_label
	12       12 -7.223970          usa
	6         6 -7.474254    indonesia
	5         5 -7.610802        india
	11       11 -7.937673           uk
	4         4 -8.004559        egypt
	13       13 -8.248504         ussr
	9         9 -8.561109  netherlands
	8         8 -8.568850       jordan
	0         0 -8.684939       brazil
	1         1 -8.820210        burma
	10       10 -9.122109       poland
	7         7 -9.267121       israel
	3         3 -9.413100         cuba
	2         2 -9.768395        china, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 53))

	Вистински релации за "usa" со "warning" релацијата:
	[]


	────────────────────────────────────────────────────────────────────────────────────────────────────

	Предвидени релации за "usa" со "weightedunvote" релацијата

	TargetPredictions(df=    tail_id      score   tail_label
	5         5 -10.074881        india
	12       12 -10.200404          usa
	7         7 -10.361886       israel
	11       11 -10.566613           uk
	8         8 -10.780188       jordan
	4         4 -10.855470        egypt
	9         9 -10.862679  netherlands
	1         1 -11.019940        burma
	6         6 -11.094213    indonesia
	10       10 -11.154152       poland
	13       13 -11.166404         ussr
	3         3 -11.268269         cuba
	2         2 -11.382254        china
	0         0 -11.623613       brazil, factory=TriplesFactory(num_entities=14, num_relations=55, create_inverse_triples=False, num_triples=201, path="C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\pykeen\\datasets\\nations\\test.txt"), target='tail', other_columns_fixed_ids=(12, 54))

	Вистински релации за "usa" со "weightedunvote" релацијата:
	['cuba', 'israel', 'ussr']


	────────────────────────────────────────────────────────────────────────────────────────────────────
	hits@1:  0.035, hits@3:  0.652, hits@10:  0.980, и MRR:  0.378


    """
