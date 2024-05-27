import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
# error cause: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
# За таа причина морав додатно матплот да го импортирам за да ми се рендираат графите на мојата околина
import tkinter
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def check_equilibrium(iterations, threshold=0.01, window=10):
    # Проверка дали промените во бројот на инфицирани се под зададениот праг во зададениот интервал
    infected_counts = [iter['node_count'][1] for iter in iterations]
    if len(infected_counts) < window:
        return False  # Недоволно податоци за проценка
    """
            'node_count' = {dict: 2} {0: 950, 1: 50}
        000 = {dict: 4} {'iteration': 0, 'node_count': {0: 950, 1: 50}, 'status': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 1, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 1, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 47: 0, 48: 0, 49: 1, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 1, 67: 0, 68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 1, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 0, 125: 1, 126: 0, 127: 0, 128: 0, 129: 0, 130: 0, 131...
    iterations_qvoter = {list: 200} [{'iteration': 0, 'node_count': {0: 950, 1: 50}, 'status': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 1, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29:
    """
    # Проверка на последните итерации
    last_counts = infected_counts[-window:]
    changes = [abs(last_counts[i] - last_counts[i - 1]) for i in range(1, len(last_counts))]
    return all(change < threshold for change in changes)


if __name__ == '__main__':
    # Дефиниција на граф од даденото множество за лабораториската
    # https://snap.stanford.edu/data/readme-Ego.txt
    g = nx.read_edgelist("facebook_combined.txt", create_using=nx.Graph(), nodetype=int)

    # Иницијализација на моделите
    voter_model = op.VoterModel(g)
    qvoter_model = op.QVoterModel(g)
    majority_model = op.MajorityRuleModel(g)
    sznajd_model = op.SznajdModel(g)

    # Конфигурација на qvoter моделите
    config_qvoter = mc.Configuration()
    config_qvoter.add_model_parameter('q',
                                      4)  # за QVoter моделот - Number of neighbours that affect the opinion of an agent
    qvoter_model.set_initial_status(config_qvoter)

    # Конфигурација на majority моделите
    config_majority = mc.Configuration()
    config_majority.add_model_parameter('q', 4)  # за majority моделот - Number of randomly chosen voters
    majority_model.set_initial_status(config_majority)

    # Извршување на симулациите
    iterations_voter = voter_model.iteration_bunch(200)
    iterations_qvoter = qvoter_model.iteration_bunch(200)
    iterations_majority = majority_model.iteration_bunch(200)
    iterations_sznajd = sznajd_model.iteration_bunch(200)

    # Изградба на трендови
    trends_voter = voter_model.build_trends(iterations_voter)
    trends_qvoter = qvoter_model.build_trends(iterations_qvoter)
    trends_majority = majority_model.build_trends(iterations_majority)
    trends_sznajd = sznajd_model.build_trends(iterations_sznajd)

    # Визуелизација
    viz_voter = DiffusionTrend(voter_model, trends_voter)
    viz_voter.plot()
    viz_qvoter = DiffusionTrend(qvoter_model, trends_qvoter)
    viz_qvoter.plot()
    viz_majority = DiffusionTrend(majority_model, trends_majority)
    viz_majority.plot()
    viz_sznajd = DiffusionTrend(sznajd_model, trends_sznajd)
    viz_sznajd.plot()

    # Употреба на функцијата за проверка на еквилибриум за секој модел
    equilibrium_voter = check_equilibrium(iterations_voter)
    equilibrium_qvoter = check_equilibrium(iterations_qvoter)
    equilibrium_majority = check_equilibrium(iterations_majority)
    equilibrium_sznajd = check_equilibrium(iterations_sznajd)

    # Печатење на резултатите за стабилноста на секој модел
    print("Voter Model Equilibrium:", equilibrium_voter)
    print("QVoter Model Equilibrium:", equilibrium_qvoter)
    print("Majority Rule Model Equilibrium:", equilibrium_majority)
    print("Sznajd Model Equilibrium:", equilibrium_sznajd)

    """
    Резултати со предходната задача се речеси исти, не гледам никаква разлика во графовите но QVoter моделот за оваа множество достига еквилибриум каде на предходната задача со случаен граф не постигаше.
    Но можно е нешто грешка да го имам подесено сето ова.
    
    OUTPUT:
    no display found. Using non-interactive Agg backend
    C:\\Users\\sQwiz\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\ndlib\\models\\DiffusionModel.py:120: UserWarning: Initial infection missing: a random sample of 5% of graph nodes will be set as infected
      warnings.warn('Initial infection missing: a random sample of 5% of graph nodes will be set as infected')
    Voter Model Equilibrium: True
    QVoter Model Equilibrium: True
    Majority Rule Model Equilibrium: False
    Sznajd Model Equilibrium: True  
    """
