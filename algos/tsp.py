from algos.base import Candidate, Population
from data.tsp_data import get_tsp_data, get_usa_map
import numpy as np
from typing import Tuple, List
import copy
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.9

POPULATION_SIZE = 200
ELITISM_RATE = 0.2
N_ITERATIONS = 1000

CITIES = get_tsp_data()


def distance_function(xy1: np.array, xy2: np.array) -> float:
    return np.sqrt(np.sum((xy1 - xy2) ** 2))


DISTANCE_MATRIX = np.array([
    [
        distance_function(city1, city2) for city2 in np.array(CITIES[["x", "y"]])
    ] for city1 in np.array(CITIES[["x", "y"]])
])


class TSPCandidate(Candidate):

    def __init__(self):
        super().__init__()
        self.chromosomes = np.arange(len(CITIES))
        # TODO : force start with a city to break symmetries
        np.random.shuffle(self.chromosomes)

    @property
    def fitness_score(self) -> float:
        total_distance = np.sum([
            DISTANCE_MATRIX[self.chromosomes[i], self.chromosomes[i+1]]
            for i in range(len(self.chromosomes) - 1)
        ]) + DISTANCE_MATRIX[self.chromosomes[-1], self.chromosomes[0]]
        return total_distance

    def mutate(self):
        if np.random.rand() > MUTATION_RATE:
            return self

        elif np.random.rand() > 0.5:
            swap = np.random.choice(np.arange(len(CITIES)), size=2, replace=False)
            self.chromosomes[swap[0]], self.chromosomes[swap[1]] = self.chromosomes[swap[1]], self.chromosomes[swap[0]]

        else:
            old_position = np.random.randint(len(CITIES))
            chromosome_to_insert = self.chromosomes[old_position]
            self.chromosomes = np.delete(self.chromosomes, old_position)

            new_position = np.random.randint(len(CITIES) - 1)
            self.chromosomes = np.insert(self.chromosomes, new_position, chromosome_to_insert)

    def crossover(self, other: "TSPCandidate", *kwargs) -> Tuple["TSPCandidate", "TSPCandidate"]:
        """
        Algorithm found here:
        https://aws.amazon.com/blogs/machine-learning/using-genetic-algorithms-on-aws-for-optimization-problems/
        """

        children_1, children_2 = copy.deepcopy(self), copy.deepcopy(other)

        if np.random.rand() > CROSSOVER_RATE:
            return children_1, children_2

        crossover_positions = np.sort(np.random.choice(np.arange(len(CITIES)), size=2, replace=False))

        for i in range(*crossover_positions):

            value_children_1 = children_1.chromosomes[i]
            value_children_2 = children_2.chromosomes[i]

            if value_children_1 == value_children_2:
                continue

            first_found_at = np.argwhere(children_1.chromosomes == value_children_1)[0]
            second_found_at = np.argwhere(children_1.chromosomes == value_children_2)[0]
            children_1.chromosomes[first_found_at], children_1.chromosomes[second_found_at] = (
                children_1.chromosomes[second_found_at], children_1.chromosomes[first_found_at]
            )

            first_found_at = np.argwhere(children_2.chromosomes == value_children_2)[0]
            second_found_at = np.argwhere(children_2.chromosomes == value_children_1)[0]
            children_2.chromosomes[first_found_at], children_2.chromosomes[second_found_at] = (
                children_2.chromosomes[second_found_at], children_2.chromosomes[first_found_at]
            )

        return children_1, children_2


class TSPPopulation(Population):

    def __init__(self):
        super().__init__(
            candidate_cls=TSPCandidate,
            population_size=POPULATION_SIZE,
            elitism_rate=ELITISM_RATE
        )


def plot_candidate(candidate: TSPCandidate):
    plt.figure()

    cities_ordered = CITIES.iloc[candidate.chromosomes]
    geometry = [Point(xy) for xy in zip(cities_ordered['longitude'], cities_ordered['latitude'])]
    gdf = gpd.GeoDataFrame(cities_ordered, geometry=geometry)

    usa_map = get_usa_map().plot()
    gdf.plot(ax=usa_map, marker='o', color='red', markersize=10)

    for i in range(len(cities_ordered) - 1):
        city_src, city_dst = cities_ordered.iloc[i], cities_ordered.iloc[i + 1]
        plt.plot(
            [city_src["longitude"], city_dst["longitude"]],
            [city_src["latitude"], city_dst["latitude"]],
            linewidth=0.5, linestyle="--", color="orange"
        )

    plt.show()



def solve():

    tracking_statistics = []

    print(f"Iteration 0 / {N_ITERATIONS}")

    population = TSPPopulation()
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

    tracking_statistics = np.array(tracking_statistics)
    x_list = np.arange(len(tracking_statistics))
    plt.plot(x_list, tracking_statistics[:, 0], label="min")
    plt.plot(x_list, tracking_statistics[:, 1], label="mean")
    plt.plot(x_list, tracking_statistics[:, 2], label="median")
    plt.plot(x_list, tracking_statistics[:, 3], label="max")

    plt.legend()
    plt.show()

    plot_candidate(best_solution)


if __name__ == "__main__":
    solve()
