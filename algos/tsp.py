import random
from functools import cached_property
import copy


STARTING_WAREHOUSE = (0, 0)

DELIVERY_STOP_LOCATIONS = [
    (1, 1),
    (4, 2),
    (5, 2),
    (6, 4),
    (4, 4),
    (3, 6),
    (1, 5),
    (2, 3),
    (8, 7),
    (8, 8),
    (1, 4),
    (4, 6),
    (5, 1),
    (1, 5),
    (9, 0),
    (1, 10),
    (10, 10),
    (3, 7),
    (9, 1),
    (5, 5)
]


class CandidateSolution(object):

    def __init__(self):
        num_stops = len(DELIVERY_STOP_LOCATIONS)
        self.path = list(range(num_stops))
        random.shuffle(self.path)


    @staticmethod
    def dist(location_a, location_b):
        xdiff = abs(location_a[0] - location_b[0])
        ydiff = abs(location_a[1] - location_b[1])
        return xdiff + ydiff

    @cached_property
    def score(self):
        # start with the distance from the warehouse to the first stop
        total_distance = self.dist(STARTING_WAREHOUSE, DELIVERY_STOP_LOCATIONS[self.path[0]])

        # then travel to each stop
        for i in range(len(self.path) - 1):
            total_distance += self.dist(
                DELIVERY_STOP_LOCATIONS[self.path[i]],
                DELIVERY_STOP_LOCATIONS[self.path[i + 1]])

        # then travel back to the warehouse
        total_distance += self.dist(STARTING_WAREHOUSE, DELIVERY_STOP_LOCATIONS[self.path[-1]])
        return total_distance


class GASolver:

    STOP_IF_NO_IMPROVEMENTS = 200

    def __init__(
        self,
        generation_size: int = 50,
        nb_iterations: int = 1000,
        elitism_rate: float = 0.05,
        crossover_rate: float = 0.30,
        tourney_size: int = 5,
        mutation_rate: float = 0.05
    ):
        self.generation_size = generation_size
        self.nb_iterations = nb_iterations
        self.elitism_rate = elitism_rate
        self.crossover_rate = crossover_rate
        self.tourney_size = tourney_size
        self.mutation_rate = mutation_rate

    def solve(self):

        best_solution, best_score = None, 1e10
        last_update = -1
        generation = []

        for i in range(self.nb_iterations):
            print(f"Iteration n°{i}")

            generation = (
                self.evolve(generation) if generation
                else [CandidateSolution() for _ in range(self.generation_size)]
            )

            generation = sorted(generation, key=lambda candidate: candidate.score)

            print(f"Min: {generation[0].score}")
            print(f"Median: {generation[len(generation) // 2].score}")
            print(f"Max: {generation[-1].score}")

            if generation[0].score < best_score:
                best_solution, best_score = generation[0], generation[0].score
                last_update = i
                print(f"New candidate found - Best score: {best_score}")

            elif i - last_update >= self.STOP_IF_NO_IMPROVEMENTS:
                break

        return best_solution


    def evolve(self, old_generation):

        elitism_size = int(self.elitism_rate * self.generation_size)
        mutation_size = int(self.mutation_rate * self.generation_size)
        selection_size = self.generation_size - elitism_size - mutation_size

        if selection_size % 2 == 1:
            elitism_size += 1

        if mutation_size % 2 == 1:
            elitism_size += 1

        # --- Elitism
        new_generation = [old_generation[i] for i in range(elitism_size)]

        # --- Selection and crossover
        for _ in range(selection_size // 2):
            # Selection
            parents = self.select_parents(old_generation, self.tourney_size)
            # Crossover
            children = self.crossover_parents_to_create_children(*parents, self.crossover_rate)
            new_generation.extend(children)

        # --- Mutation
        for _ in range(mutation_size // 2):
            candidate = random.choice(old_generation)
            new_generation.append(self.swap_mutation(candidate))
            old_generation.append(self.displacement_mutation(candidate))

        return new_generation

    @staticmethod
    def tourney_select(generation, tourney_size):
        selected = random.sample(generation, tourney_size)
        best = min(selected, key=lambda candidate: candidate.score)
        return best

    @staticmethod
    def select_parents(generation, tourney_size):
        # using Tourney selection, get two candidates and make sure they're distinct
        candidate1, candidate2 = -1, 1
        while candidate1 != candidate2:
            candidate1 = GASolver.tourney_select(generation, tourney_size)
            candidate2 = GASolver.tourney_select(generation, tourney_size)
        return candidate1, candidate2

    @staticmethod
    def crossover_parents_to_create_children(parent_1, parent_2, crossover_rate):
        child1 = copy.deepcopy(parent_1)
        child2 = copy.deepcopy(parent_2)

        # sometimes we don't cross over, so use copies of the parents
        if random.random() >= crossover_rate:
            return child1, child2

        num_genes = len(parent_1.path)

        # pick a point between 0 and the end - 2, so we can cross at least 1 stop
        start_cross_at = random.randint(0, num_genes - 2)
        num_remaining = num_genes - start_cross_at
        end_cross_at = random.randint(num_genes - num_remaining + 1, num_genes - 1)

        for index in range(start_cross_at, end_cross_at + 1):
            child1_stop = child1.path[index]
            child2_stop = child2.path[index]

            # if the same, skip it since there is no crossover needed at this gene
            if child1_stop == child2_stop:
                continue

            # find within child1 and swap
            first_found_at = child1.path.index(child1_stop)
            second_found_at = child1.path.index(child2_stop)
            child1.path[first_found_at], child1.path[second_found_at] = child1.path[second_found_at], child1.path[
                first_found_at]

            # and the same for the second child
            first_found_at = child2.path.index(child1_stop)
            second_found_at = child2.path.index(child2_stop)
            child2.path[first_found_at], child2.path[second_found_at] = child2.path[second_found_at], child2.path[
                first_found_at]

        return child1, child2

    @staticmethod
    def swap_mutation(candidate):
        candidate = copy.deepcopy(candidate)
        indexes = range(len(candidate.path))
        pos1, pos2 = random.sample(indexes, 2)
        candidate.path[pos1], candidate.path[pos2] = candidate.path[pos2], candidate.path[pos1]
        return candidate

    @staticmethod
    def displacement_mutation(candidate):
        candidate = copy.deepcopy(candidate)
        num_stops = len(candidate.path)
        stop_to_move = random.randint(0, num_stops - 1)
        insert_at = random.randint(0, num_stops - 1)
        # make sure it's moved to a new index within the path, so it's really different
        while insert_at == stop_to_move:
            insert_at = random.randint(0, num_stops - 1)
        stop_index = candidate.path[stop_to_move]
        del candidate.path[stop_to_move]
        candidate.path.insert(insert_at, stop_index)
        return candidate


if __name__ == "__main__":

    solver = GASolver()
    solver.solve()