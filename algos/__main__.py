import logging
import click

from algos.one_max import solve_one_max
from algos.knapsack import solve_knapsack
from algos.continuous_optimization import solve_continuous_optim
from algos.tsp import solve_tsp

logging.basicConfig(level=logging.INFO)

SUPPORTED_ALGOS = [
    ("continuous_optim", solve_continuous_optim),
    ("knapsack", solve_knapsack),
    ("one_max", solve_one_max),
    ("tsp", solve_tsp),
]


@click.command()
@click.option("--algorithm", type=click.Choice([algo[0] for algo in SUPPORTED_ALGOS]), required=True)
def main(algorithm):
    algo_to_call = next(algo[1] for algo in SUPPORTED_ALGOS if algo[0] == algorithm)
    algo_to_call()


if __name__ == "__main__":
    main()
