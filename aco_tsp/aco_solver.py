from typing import List
import numpy as np
from aco_tsp.ant import Ant
from tqdm import tqdm


class ACOSolver:

    def __init__(self,
                 n_ants: int,
                 rho: float,
                 alpha: float,
                 beta: float,
                 pheromones_matrix: np.ndarray,
                 desirability_matrix: np.ndarray):
        self.n_ants = n_ants
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.pheromones_matrix = pheromones_matrix
        self.desirability_matrix = desirability_matrix

        self.solutions = []
        self.fitnesses = []

        self.best_solution = None
        self.best_fitness = np.inf

        self.best_solutions = []
        self.best_fitnesses = []

    def _create_ant_colony(self, available_cities: List[int]) -> List[Ant]:
        return [Ant(self.alpha, self.beta, available_cities) for _ in range(self.n_ants)]

    def _update_pheromone_matrix(self):
        update_pheromone_matrix = np.zeros(self.pheromones_matrix.shape)
        for solution, path_fitness in zip(self.solutions, self.fitnesses):
            pheromone_update = 1. / path_fitness
            for edge in zip(solution[:-1], solution[1:]):
                update_pheromone_matrix[edge[0], edge[1]] += pheromone_update
        self.pheromones_matrix = (1 - self.rho) * self.pheromones_matrix + update_pheromone_matrix

    def _ant_search(self, available_cities: List[int]):
        ants = self._create_ant_colony(available_cities)

        for ant in ants:
            ant.solve(self.pheromones_matrix, self.desirability_matrix)
            self.solutions.append(ant.get_solution_path())
            self.fitnesses.append(ant.get_solution_fitness())

    def run(self, max_iter: int, available_cities: List[int]):
        print(f"Searching solutions for {max_iter} iters")
        for _ in tqdm(range(max_iter)):
            self._ant_search(available_cities)
            self.get_best_solution()
            self._update_pheromone_matrix()

            self.solutions = []
            self.fitnesses = []
        print("Best solution:", self.best_solution)
        print("Best fitness: ", self.best_fitness)

    def get_best_solution(self):
        index_best = np.argmin(self.fitnesses)
        best_fitness = self.fitnesses[index_best]
        best_solution = self.solutions[index_best]

        if best_fitness < self.best_fitness:
            self.best_fitness = best_fitness
            self.best_solution = best_solution
            self.best_fitnesses.append(self.best_fitness)
            self.best_solutions.append(self.best_solution)

        return self.best_fitness, self.best_solution

    def get_best_solutions_history(self):
        return self.best_solutions
