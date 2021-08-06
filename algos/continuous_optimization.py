from algos.base import Candidate, Population
import numpy as np
from typing import Tuple, List
import copy
import inspect
import matplotlib.pyplot as plt


def rastrigni_function(x1: float, x2: float) -> float:
    return (
        20 + x1 ** 2 + x2 ** 2 - 10 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))
    )


def basic_quadratic_function(x: float) -> float:
    return x ** 2


FUNCTION_TO_OPTIMIZE = (
    rastrigni_function
    # basic_quadratic_function
)
N_PARAMETERS = len(inspect.signature(FUNCTION_TO_OPTIMIZE).parameters)
BOUNDS = [-10, 10]

MUTATION_RATE = 0.35
CROSSOVER_RATE = 0.9

BETA_CROSSOVER = 0.5

POPULATION_SIZE = 50
ELITISM_RATE = 0.2

N_ITERATIONS = 200


class ContinuousOptimCandidate(Candidate):

    def __init__(self):
        super().__init__()
        self.chromosomes = np.random.uniform(*BOUNDS, N_PARAMETERS)
        # For the sake of the exercise, try to start, far away from the solution
        self.chromosomes = np.random.uniform(BOUNDS[0], BOUNDS[0] + 7, N_PARAMETERS)

    @property
    def fitness_score(self) -> float:
        return FUNCTION_TO_OPTIMIZE(*self.chromosomes)

    def mutate(self):
        if np.random.rand() > MUTATION_RATE:
            return self

        mutation_point = np.random.randint(N_PARAMETERS)

        self.chromosomes[mutation_point] += np.random.uniform(
            BOUNDS[0] - self.chromosomes[mutation_point],
            BOUNDS[1] - self.chromosomes[mutation_point]
        ) * MUTATION_RATE

    def crossover(self, other: "ContinuousOptimCandidate", *kwargs) -> Tuple[
        "ContinuousOptimCandidate", "ContinuousOptimCandidate"
    ]:

        children_1, children_2 = copy.deepcopy(self), copy.deepcopy(other)

        if np.random.rand() > CROSSOVER_RATE:
            return children_1, children_2

        crossover_point = np.random.randint(N_PARAMETERS)
        children_1.chromosomes[crossover_point] = (
            BETA_CROSSOVER * children_1.chromosomes[crossover_point]
            + (1 - BETA_CROSSOVER) * other.chromosomes[crossover_point]
        )
        children_2.chromosomes[crossover_point] = (
            BETA_CROSSOVER * children_2.chromosomes[crossover_point]
            + (1 - BETA_CROSSOVER) * self.chromosomes[crossover_point]
        )

        return children_1, children_2


class ContinuousOptimPopulation(Population):

    def __init__(self):
        super().__init__(
            candidate_cls=ContinuousOptimCandidate,
            population_size=POPULATION_SIZE,
            elitism_rate=ELITISM_RATE
        )


def solve():

    tracking_statistics = []
    tracking_best_solution = []

    print(f"Iteration 0 / {N_ITERATIONS}")

    population = ContinuousOptimPopulation()
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
            tracking_best_solution.append(best_solution)

    print(best_solution.chromosomes)

    plt.figure()
    tracking_statistics = np.array(tracking_statistics)
    x_list = np.arange(len(tracking_statistics))
    plt.plot(x_list, tracking_statistics[:, 0], label="min")
    plt.plot(x_list, tracking_statistics[:, 1], label="mean")
    plt.plot(x_list, tracking_statistics[:, 2], label="median")
    plt.plot(x_list, tracking_statistics[:, 3], label="max")
    plt.legend()
    plt.show()

    if N_PARAMETERS == 2:
        plt.figure()
        x, y = np.linspace(*BOUNDS, 500), np.linspace(*BOUNDS, 500)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(FUNCTION_TO_OPTIMIZE)(X, Y)
        plt.pcolor(X, Y, Z)

        tracking_best = np.array([c.chromosomes for c in tracking_best_solution])
        plt.scatter(tracking_best[:, 0], tracking_best[:, 1], color="red")
        plt.scatter(tracking_best[-1, 0], tracking_best[-1, 1], color="green")
        plt.show()


if __name__ == "__main__":
    solve()
