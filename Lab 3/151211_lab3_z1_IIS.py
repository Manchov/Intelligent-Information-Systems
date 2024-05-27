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
        000 = {dict: 4} {'iteration': 0, 'node_count': {0: 950, 1: 50}, 'status': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 1, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 1, 47: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 1, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 1, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0, 68: 0, 69: 0, 70: 1, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 1, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 0, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 92: 0, 93: 0, 94: 0, 95: 0, 96: 0, 97: 0, 98: 0, 99: 0, 100: 0, 101: 0, 102: 0, 103: 0, 104: 0, 105: 0, 106: 0, 107: 0, 108: 0, 109: 0, 110: 0, 111: 0, 112: 0, 113: 1, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 119: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 1, 125: 0, 126: 1, 127: 1, 128: 0, 129: 0, 130: 0, 131...
    iterations_si = {list: 200} [{'iteration': 0, 'node_count': {0: 950, 1: 50}, 'status': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29:
    """
    # Проверка на последните итерации
    last_counts = infected_counts[-window:]
    changes = [abs(last_counts[i] - last_counts[i - 1]) for i in range(1, len(last_counts))]
    return all(change < threshold for change in changes)


if __name__ == '__main__':
    # Дефинирање на мрежа
    g = nx.erdos_renyi_graph(1000, 0.1)

    # Иницијализација на моделите
    si_model = ep.SIModel(g)
    sis_model = ep.SISModel(g)
    seis_model = ep.SEISModel(g)
    seir_model = ep.SEIRModel(g)

    # Конфигурација на SI моделите
    config_si = mc.Configuration()
    config_si.add_model_parameter('beta', 0.01)  # Стапка на инфекција
    config_si.add_model_parameter("fraction_infected", 0.05)  # Фракција на инфицирани
    si_model.set_initial_status(config_si)

    # Конфигурација на SIS моделот
    config_sis = mc.Configuration()
    config_sis.add_model_parameter('beta', 0.01)
    config_sis.add_model_parameter('lambda', 0.005)  # Стапка на повторна инфекција
    config_sis.add_model_parameter("fraction_infected", 0.05)
    sis_model.set_initial_status(config_sis)

    # Конфигурација на SEIS моделот
    config_seis = mc.Configuration()
    config_seis.add_model_parameter('alpha', 0.05)  # Стапка на прогресија од изложеност кон инфекција
    config_seis.add_model_parameter('beta', 0.01)
    config_seis.add_model_parameter('lambda', 0.005)  # Стапка на повторна инфекција
    config_seis.add_model_parameter("fraction_infected", 0.05)
    seis_model.set_initial_status(config_seis)

    # Конфигурација на SEIR моделот
    config_seir = mc.Configuration()
    config_seir.add_model_parameter('alpha', 0.05)  # Инкубационен период
    config_seir.add_model_parameter('beta', 0.01)
    config_seir.add_model_parameter('gamma', 0.01)  # Стапка на опоравување
    config_seir.add_model_parameter("fraction_infected", 0.05)
    seir_model.set_initial_status(config_seir)

    # Симулација
    iterations_si = si_model.iteration_bunch(200)
    iterations_sis = sis_model.iteration_bunch(200)
    iterations_seis = seis_model.iteration_bunch(200)
    iterations_seir = seir_model.iteration_bunch(200)

    # Изградба на трендови
    trends_si = si_model.build_trends(iterations_si)
    trends_sis = sis_model.build_trends(iterations_sis)
    trends_seis = seis_model.build_trends(iterations_seis)
    trends_seir = seir_model.build_trends(iterations_seir)

    # Визуелизација
    viz_si = DiffusionTrend(si_model, trends_si)
    viz_si.plot()
    viz_sis = DiffusionTrend(sis_model, trends_sis)
    viz_sis.plot()
    viz_seis = DiffusionTrend(seis_model, trends_seis)
    viz_seis.plot()
    viz_seir = DiffusionTrend(seir_model, trends_seir)
    viz_seir.plot()

    # Употреба на функцијата за проверка на еквилибриум за секој модел
    equilibrium_si = check_equilibrium(iterations_si)
    equilibrium_sis = check_equilibrium(iterations_sis)
    equilibrium_seis = check_equilibrium(iterations_seis)
    equilibrium_seir = check_equilibrium(iterations_seir)

    # Печатење на резултатите за стабилноста на секој модел
    print("SI Model Equilibrium:", equilibrium_si)
    print("SIS Model Equilibrium:", equilibrium_sis)
    print("SEIS Model Equilibrium:", equilibrium_seis)
    print("SEIR Model Equilibrium:", equilibrium_seir)

    """
    OUTPUT:
    no display found. Using non-interactive Agg backend
    SI Model Equilibrium: True
    SIS Model Equilibrium: False
    SEIS Model Equilibrium: False
    SEIR Model Equilibrium: False

    Process finished with exit code 0
    """
