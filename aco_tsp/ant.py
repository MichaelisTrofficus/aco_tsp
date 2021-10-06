from typing import List

import numpy as np

from aco_tsp.utils import roulette_wheel


class Ant:
    """
    Class defining a computational ant
    """
    def __init__(self,
                 alpha: float,
                 beta: float,
                 available_cities: List[int]):

        self.alpha = alpha
        self.beta = beta
        self.available_cities = available_cities.copy()

        self.current_position = 0
        self.available_cities.remove(self.current_position)

        self.path = [self.current_position]
        self.solution_fitness = 0.

    def _get_total_prob(self, pheromones_matrix, desirability_matrix):
        """
        :return:
        """
        total_prob = 0.
        for city in self.available_cities:
            arc_pheromones = pheromones_matrix[self.current_position, city] ** self.alpha
            arc_desirability = desirability_matrix[self.current_position, city] ** self.beta
            total_prob += arc_pheromones * arc_desirability
        return total_prob

    def _select_next_move(self, pheromones_matrix: np.ndarray, desirability_matrix: np.ndarray):
        """

        :param pheromones_matrix:
        :param desirability_matrix:
        :return:
        """
        probs = []
        denominator = self._get_total_prob(pheromones_matrix, desirability_matrix)

        for city in self.available_cities:
            arc_pheromones = pheromones_matrix[self.current_position, city]**self.alpha
            arc_desirability = desirability_matrix[self.current_position, city]**self.beta
            probs.append((arc_pheromones*arc_desirability) / denominator)

        next_city_index = roulette_wheel(probs)
        next_city = self.available_cities[next_city_index]

        self.available_cities.remove(next_city)
        self.path.append(next_city)
        self.solution_fitness += 1./desirability_matrix[self.current_position, next_city]

        self.current_position = next_city

    def solve(self, pheromones_matrix: np.ndarray, desirability_matrix: np.ndarray):
        while self.available_cities:
            self._select_next_move(pheromones_matrix, desirability_matrix)
        # We add the final return to the nest
        self.solution_fitness += 1./desirability_matrix[self.current_position, 0]
        self.path.append(0)

    def get_solution_path(self):
        return self.path

    def get_solution_fitness(self):
        return self.solution_fitness
