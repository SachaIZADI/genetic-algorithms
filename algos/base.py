from abc import ABC
from functools import cached_property
from typing import Tuple, List
import numpy as np


class Candidate(ABC):

    @cached_property
    def fitness_score(self) -> float:
        return None

    def mutate(self):
        pass

    def crossover(self, other: "Candidate", *kwargs) -> Tuple["Candidate", "Candidate"]:
        return self, other

    def crossover_and_mutate(self, other: "Candidate", *kwargs) -> Tuple["Candidate", "Candidate"]:
        children = self.crossover(other, *kwargs)
        children[0].mutate()
        children[1].mutate()
        return children


class Population(ABC):

    def __init__(
        self,
        candidate_cls,
        population_size: int,
        elitism_rate: float,
        *kwargs
    ):
        assert population_size % 2 == 0

        self.population_size = population_size
        self.elite_size = int(2 * ((elitism_rate * population_size) // 2))

        self.candidates = [candidate_cls() for _ in range(population_size)]

    def evolve(self):
        self.sort()
        new_candidates = []

        # Elitism
        new_candidates.extend(self.select_elite())

        # Selection & Crossover & Mutation
        for _ in range((self.population_size - self.elite_size) // 2):
            parents = self.select_parents()
            children = parents[0].crossover_and_mutate(parents[1])
            new_candidates.extend(children)

        self.candidates = new_candidates

    def select_elite(self) -> List[Candidate]:
        return self.candidates[: self.elite_size]

    def select_parents(self) -> Tuple[Candidate, Candidate]:
        fitness_scores = np.array([candidate.fitness_score for candidate in self.candidates])
        selection_probability = fitness_scores / fitness_scores.sum()
        return tuple(np.random.choice(self.candidates, 2, p=selection_probability))

    def sort(self):
        self.candidates = sorted(self.candidates, key=lambda c: c.fitness_score)

    @property
    def statistics(self) -> Tuple[float, float, float, float]:
        fitness_scores = np.array([candidate.fitness_score for candidate in self.candidates])
        return (
            fitness_scores.min(),
            fitness_scores.mean(),
            np.median(fitness_scores),
            fitness_scores.max()
        )

