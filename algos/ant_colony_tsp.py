import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import geopandas as gpd
from shapely.geometry import Point

from data.tsp_data import get_tsp_data, get_usa_map


def plot_path(path: List[int]):
    plt.figure()
    cities = get_tsp_data()

    cities_ordered = cities.iloc[path + [0]]
    geometry = [Point(xy) for xy in zip(cities_ordered['longitude'], cities_ordered['latitude'])]
    gdf = gpd.GeoDataFrame(cities_ordered, geometry=geometry)

    usa_map = get_usa_map().plot()
    gdf.plot(ax=usa_map, marker='o', color='red', markersize=10)

    for i in range(len(cities_ordered) - 1):
        city_src, city_dst = cities_ordered.iloc[i], cities_ordered.iloc[i + 1]
        plt.plot(
            [city_src["longitude"], city_dst["longitude"]],
            [city_src["latitude"], city_dst["latitude"]],
            linewidth=2, linestyle="--", color="orange"
        )

    plt.show()


class CityGraph(nx.DiGraph):

    def __init__(self):
        super().__init__()
        self._load_data()

    def _load_data(self):
        tsp_data = get_tsp_data()
        for i in range(len(tsp_data)):
            self.add_node(i, **tsp_data.iloc[i].to_dict())

        for node_1 in self.nodes:
            for node_2 in self.nodes:

                if node_1 == node_2:
                    continue

                distance = self.compute_distance_between_cities(
                    np.array([self.nodes[node_1]["x"], self.nodes[node_1]["y"]]),
                    np.array([self.nodes[node_2]["x"], self.nodes[node_2]["y"]]),
                )

                self.add_edge(
                    node_1,
                    node_2,
                    distance=distance,
                    pheromone=1,
                    selection_factor=1
                )

    @staticmethod
    def compute_distance_between_cities(xy1: np.array, xy2: np.array) -> float:
        return np.sqrt(np.sum((xy1 - xy2) ** 2))

    def global_update_pheromone(self, rho: float):
        for edge in self.edges:
            self.edges[edge]["pheromone"] = (1 - rho) * self.edges[edge]["pheromone"]

    def global_update_selection_factor(self, alpha: float, beta: float, gamma: float):
        for edge in self.edges:
            self.edges[edge]["selection_factor"] = (
                gamma
                + (
                        self.edges[edge]["pheromone"] ** alpha
                        * (1 / self.edges[edge]["distance"]) ** beta
                )
            )

    def select_next_node(self, current_node: int, nodes_to_visit: List[int]):
        if len(nodes_to_visit) == 1:
            return nodes_to_visit[0]

        probabilities = np.array([
            self.edges[(current_node, node_to_visit)]["selection_factor"]
            for node_to_visit in nodes_to_visit
        ])

        probabilities = probabilities / np.sum(probabilities)
        chosen_node = np.random.choice(nodes_to_visit, p=probabilities)

        return chosen_node


class Ant(object):

    def __init__(self, graph: CityGraph, Q: float = 0.1):
        self.graph = graph
        self.Q = Q

        self.path = []
        self.position = None
        self.nodes_to_visit = []

    def initialize(self):
        self.path = [0]
        self.position = 0
        self.nodes_to_visit = list(self.graph.nodes)
        self.nodes_to_visit.remove(0)

    @property
    def distance(self) -> float:
        return np.sum([
            self.graph.edges[(self.path[i], self.path[i+1])]["distance"]
            for i in range(len(self.path) - 1)
        ]) + self.graph.edges[(self.path[-1], 0)]["distance"]

    def go_to_next_node(self):
        next_node = self.graph.select_next_node(
            current_node=self.position,
            nodes_to_visit=self.nodes_to_visit
        )
        self.path.append(next_node)
        self.position = next_node
        self.nodes_to_visit.remove(next_node)

    def local_update_pheromone(self,):
        for i in range(len(self.path) - 1):
            self.graph.edges[(self.path[i], self.path[i+1])]["pheromone"] += self.Q / self.distance

        self.graph.edges[(self.path[-1], 0)]["pheromone"] += self.Q / self.distance

    def explore_graph(self):
        self.initialize()
        while self.nodes_to_visit:
            self.go_to_next_node()
        self.local_update_pheromone()


class AntColonyOptimization(object):

    N_ITERATIONS: int = 200
    NB_ANTS: int = 30

    ALPHA: float = 1
    BETA: float = 1
    GAMMA: float = 0.1
    Q: float = 5e7
    RHO: float = 0.1

    def __init__(
        self,
        city_graph: CityGraph,
    ):
        self.city_graph = city_graph

        # Compute an order of magnitude of the quantity of pheromone left by each ant so that it is not too low
        self.Q = np.sum([
            city_graph.edges[(i, i+1)]["distance"]
            for i in range(len(city_graph) - 1)
        ]) * 100

        self.ants = [Ant(graph=city_graph, Q=self.Q) for _ in range(self.NB_ANTS)]


    @property
    def tracking_statistic(self) -> Tuple[float, float, float]:
        distances = [ant.distance for ant in self.ants]
        return (
            np.min(distances),
            np.mean(distances),
            np.median(distances),
            np.max(distances),
        )

    def solve(self):

        best_path = []
        best_score = np.inf
        tracking_statistics = []

        for i in range(self.N_ITERATIONS):

            print(f"Iteration {i + 1} / {self.N_ITERATIONS}")

            self.city_graph.global_update_pheromone(rho=self.RHO)

            for ant in self.ants:
                ant.explore_graph()

                if ant.distance < best_score:
                    best_score = ant.distance
                    best_path = ant.path

            self.city_graph.global_update_selection_factor(
                alpha=self.ALPHA,
                beta=self.BETA,
                gamma=self.GAMMA,
            )

            tracking_statistics.append((best_score, *self.tracking_statistic))

        tracking_statistics = np.array(tracking_statistics)
        x_list = np.arange(len(tracking_statistics))
        plt.plot(x_list, tracking_statistics[:, 0], label="best")
        plt.plot(x_list, tracking_statistics[:, 1], label="min")
        plt.plot(x_list, tracking_statistics[:, 2], label="mean")
        plt.plot(x_list, tracking_statistics[:, 3], label="median")
        plt.plot(x_list, tracking_statistics[:, 4], label="max")
        plt.legend()
        plt.show()

        plot_path(best_path)


def solve_tsp_aco():
    city_graph = CityGraph()
    AntColonyOptimization(city_graph=city_graph).solve()


if __name__ == "__main__":
    solve_tsp_aco()
