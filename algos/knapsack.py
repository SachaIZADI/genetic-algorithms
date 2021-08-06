from algos.base import Candidate, Population
import numpy as np
from typing import Tuple, List
import copy
import matplotlib.pyplot as plt

ITEMS = np.array([
    # (weight, value)
    (10, 100),
    (5, 60),
    (1, 50),
    (5, 55),
    (2, 20),
    (9, 100),
    (1, 10),
    (1, 20),
    (3, 10),
])
CAPACITY = 20

MUTATION_RATE = 0.005
CROSSOVER_RATE = 0.9

POPULATION_SIZE = 10
ELITISM_RATE = 0.2

N_ITERATIONS = 200


class KnapSackCandidate(Candidate):

    def __init__(self):
        super().__init__()
        self.chromosomes = np.random.randint(0, 2, len(ITEMS))

    @property
    def weight(self) -> float:
        return (ITEMS[:, 0] * self.chromosomes).sum()

    @property
    def fitness_score(self) -> float:
        if self.weight > CAPACITY:
            return 0
        return - (ITEMS[:, 1] * self.chromosomes).sum()

    def mutate(self):
        mutations = np.random.binomial(size=len(ITEMS), n=1, p=MUTATION_RATE)
        self.chromosomes = np.logical_xor(self.chromosomes, mutations).astype(int)

    def crossover(self, other: "KnapSackCandidate", *kwargs) -> Tuple["KnapSackCandidate", "KnapSackCandidate"]:

        children_1, children_2 = copy.deepcopy(self), copy.deepcopy(other)

        if np.random.rand() > CROSSOVER_RATE:
            return children_1, children_2

        crossover_point = np.random.randint(1, len(ITEMS))
        children_1.chromosomes[crossover_point: -1] = other.chromosomes[crossover_point: -1]
        children_2.chromosomes[crossover_point: -1] = self.chromosomes[crossover_point: -1]

        return children_1, children_2


class KnapSackPopulation(Population):

    def __init__(self):
        super().__init__(
            candidate_cls=KnapSackCandidate,
            population_size=POPULATION_SIZE,
            elitism_rate=ELITISM_RATE
        )


def solve_knapsack():

    tracking_statistics = []

    print(f"Iteration 0 / {N_ITERATIONS}")

    population = KnapSackPopulation()
    print(population.statistics)

    best_solution = min(population.candidates, key=lambda c: c.fitness_score)
    best_score = best_solution.fitness_score
    tracking_statistics.append(population.statistics)

    for i in range(N_ITERATIONS):
        print(f"Iteration {i + 1} / {N_ITERATIONS}")
        population.evolve()

        print(population.statistics)
        tracking_statistics.append(population.statistics)

        if population.candidates[0].fitness_score < best_score:
            best_solution = population.candidates[0]
            best_score = best_solution.fitness_score

    print(best_solution.chromosomes)
    print(f"Used capacity {best_solution.weight}")

    tracking_statistics = np.array(tracking_statistics)
    x_list = np.arange(len(tracking_statistics))
    plt.plot(x_list, tracking_statistics[:, 0], label="min")
    plt.plot(x_list, tracking_statistics[:, 1], label="mean")
    plt.plot(x_list, tracking_statistics[:, 2], label="median")
    plt.plot(x_list, tracking_statistics[:, 3], label="max")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    solve_knapsack()

