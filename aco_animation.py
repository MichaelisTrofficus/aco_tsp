import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from aco_tsp.utils import parse_problem_instance, compute_cost_matrix, compute_pheromones_matrix
from aco_tsp.aco_solver import ACOSolver


def aco(problem_instance_path: str,
        n_ants: int = 100,
        rho: float = 0.1,
        alpha: float = 0.6,
        beta: float = 3.0,
        max_iter: int = 500):
    """

    :param problem_instance_path:
    :param n_ants: Number of ants
    :param rho: Parameter to control evaporation
    :param alpha: Parameter to regulate pheromone attraction
    :param beta: Parameter to regulate desirability
    :param max_iter: Maximum number of iterations
    :return:
    """

    # PROBLEM DEFINITION
    cities_coord, label, color_label = parse_problem_instance(problem_instance_path)

    # SOLVE THE TSP PROBLEM
    cost_matrix = compute_cost_matrix(cities_coord)
    pheromones_matrix = compute_pheromones_matrix(cities_coord, cost_matrix)
    desirability_matrix = 1. / cost_matrix

    aco_solver = ACOSolver(n_ants=n_ants, rho=rho, alpha=alpha, beta=beta,
                           pheromones_matrix=pheromones_matrix, desirability_matrix=desirability_matrix)

    aco_solver.run(max_iter, available_cities=label)

    best_solutions = aco_solver.get_best_solutions_history()

    # VISUALIZATION

    fig, ax = plt.subplots()
    plt.scatter(cities_coord[0], cities_coord[1], c=color_label)
    ln, = plt.plot([], [], "b--")

    def init():
        return ln,

    def update(frame):
        ln.set_data(list(cities_coord[0][frame]), list(cities_coord[1][frame]))
        return ln,

    _ = FuncAnimation(fig, update, frames=best_solutions,
                      init_func=init, blit=True, repeat=False,
                      interval=1000, repeat_delay=10000)
    plt.show()
