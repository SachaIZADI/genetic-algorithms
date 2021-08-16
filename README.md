# Genetic Algorithms

## Some resources

- A good and quite complete introduction to genetic algorithms: 
  https://www.whitman.edu/Documents/Academics/Mathematics/2014/carrjk.pdf
  
- A nice step-by-step tutorial on genetic algorithms applied to the travelling salesman problem: 
  https://aws.amazon.com/blogs/machine-learning/using-genetic-algorithms-on-aws-for-optimization-problems/
  
- A theoretical proof of the convergence of genetic algorithms:
  https://www.imo.universite-paris-saclay.fr/~cerf/papers/gae.pdf
  
## Implementations

You can call the following algorithms with the CLI interface :
```
python -m algos --algorithm [continuous_optim|knapsack|one_max|tsp]
```

For the TSP you can run a dedicated steamlit app via:
```
streamlit run tsp_streamlit_app.py 
```

![streamlit_app](media/streamlit_tsp.gif)

### OneMax problem

The OneMax problem is the following (with i between 1 and 20):

![one_max_equation](media/one_max_equation.png)

which is obviously maximised for `a_i = 1` with `maximum = 20`.

We see that the algorithm progressively decreases the objective function and converges to the solution.

![one_max_solution](media/one_max.png)

### Knapsack problem

The Knapsack problem problem is the following:

![knapsack_equation](media/knapsack_equation.png)

To handle the constraint on the capacity, we transform the objective function such that it is equal to zero if the
constraint is not satisfied.

![knapsack_equation](media/knapsack_equation_2.png)

We see that the algorithm progressively decreases the objective function and converges to the solution, but it keeps
exploring parts of the solution space where the constraint is not verified.

![knapsack_solution](media/knapsack.png)

### Continuous optimization

The problem is to find the (global) minimum of the Rastrigni function. It has a lot of local minimum which makes the
problem interesting.

![rastrigni_equation](media/Rastrigin_function.png)

For the crossover operation, I used the trick presented in https://www.whitman.edu/Documents/Academics/Mathematics/2014/carrjk.pdf

```
x_child_1 = (1 − β) * x_parent_1 + β * x_parent_2
x_child_2 = (1 − β) * x_parent_2 + β * x_parent_1
```

We see that the algorithm progressively decreases the objective function and explores several local minima until
finding the global minimum.

![continuous_optim_solution](media/continuous_optim.png)

![continuous_optim_solution_2](media/continuous_optim_2.png)

### Travelling salesman problem

The TSP problem is to minimize the total distance to do a round trip visiting each capital of the US.
We see that the algorithm progressively decreases the objective function but seems to converge to a local optimum.

The solution found has still many defaults (esp. crossing roads) but it mostly has "continuous" paths. 
To improve our algorithm we should rewrite the model formulation (e.g. breaking the symmetries) and create other
mutation/crossover strategies to make the solution space exploration more efficient.

![tsp_solution](media/tsp.png)

![tsp_solution_2](media/tsp_solution.jpeg)


## Bonus part - Ant colony optimization

To go a bit further in the exploration of bio-inspired optimization techniques I also implemented some ant colony
optimization algorithms.

The Wikipedia article on [ACO](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms) is a good introduction 
to the subject.

### Travelling salesman problem

ACO techniques apply to optimization problems that can be formulated as finding the shortest path on a weighted graph,
which is exactly the TSP formulation.

ACO gave better results, and in a much smaller amount of time, than the genetic algorithm implemented earlier on. 
Especially the tour looks quite cleaner than the genetic algorithm which had several inefficient back-and-forth.

![aco_tsp_solution_2](media/aco_tsp.png)

![aco_tsp_solution](media/aco_tsp_map.png)

![tsp_solution_2](media/tsp_solution.jpeg)



