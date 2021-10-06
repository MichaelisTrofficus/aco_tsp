from typing import Tuple, List
import numpy as np


def parse_problem_instance(path: str) -> Tuple[np.ndarray, List[int], List[str]]:
    with open(f"{path}", "r") as f:
        data = f.readlines()
    x_coord = [int(i) for i in data[0].split(",")]
    y_coord = [int(j) for j in data[1].split(",")]
    label = [x for x in range(len(x_coord))]
    color_label = ["r"] + ["b"]*(len(x_coord) - 1)
    return np.array([x_coord, y_coord]), label, color_label


def compute_cost_matrix(cities_coord: np.ndarray) -> np.ndarray:
    """
    Computes the cost matrix for the problem.
    :param cities_coord: (x, y) coordinates for the cities
    :return: The cost matrix
    """
    cities_coord = np.transpose(cities_coord)
    n_cities = cities_coord.shape[0]
    cost_matrix = np.zeros([n_cities, n_cities])

    for i in range(n_cities):
        for j in range(n_cities):
            cost_matrix[i, j] = np.sqrt(
                (cities_coord[i, 0] - cities_coord[j, 0])**2 +
                (cities_coord[i, 1] - cities_coord[j, 1])**2)
    return cost_matrix


def compute_pheromones_matrix(cities_coord: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the initial pheromones matrix. It adds some initial pheromones levels.
    :param cities_coord:
    :param cost_matrix:
    :return:
    """
    cities_coord = np.transpose(cities_coord)
    n_cities = cities_coord.shape[0]
    pheromones_matrix = np.ones([n_cities, n_cities])

    # We add some initial pheromones
    pheromone_initial_level = 10 * (1 / n_cities * np.mean(cost_matrix))
    return pheromones_matrix * pheromone_initial_level


def roulette_wheel(probs: List[float]) -> int:
    """
    :param probs:
    :return:
    """
    probs_cumsum = np.cumsum(probs)
    random_number = np.random.random()
    return int(np.argmax(random_number <= probs_cumsum))

