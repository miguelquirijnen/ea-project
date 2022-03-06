from typing import List
from numba.core.types.scalars import Boolean
from reporter import Reporter
import numpy as np
import copy
import random
import numpy as np
import itertools
from math import isinf
from numba import jit


class r0710304:

    def __init__(self):
        self.reporter = Reporter(self.__class__.__name__)

        # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        timeLeft = 300
        solver = Solver(distanceMatrix, timeLeft)
        population = solver.initialize(distanceMatrix)

        while solver.should_continue(timeLeft):

            # Your code here.
            population = solver.evolutionary_iteration(
                distanceMatrix, population)

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(
                solver.mean, solver.best, solver.bestRoute)
            if timeLeft < 0:
                break

            solver.timeleft = timeLeft

        # Your code here.
        return solver.best, solver.mean


class Individual:
    def __init__(self, distanceMatrix, route=None):
        self.fitness = -1
        self.id = random.randint(0, 10000)

        self.sigma = 2

        if route is None:
            self.route = list(np.random.permutation(len(distanceMatrix)))
        else:
            self.route = route

        self.computeFitness(distanceMatrix)

    def computeFitness(self, distanceMatrix: np.array):
        """
        Computes the fitness by summing the distances in the route
        """
        self.fitness = 0
        for i in range(len(self.route)):
            dist = distanceMatrix[self.route[i]
                                  ][self.route[(i+1) % len(self.route)]]
            self.fitness += dist

        if (isinf(self.fitness)):
            self.fitness = 2e+100

        return self.fitness

    def __repr__(self):
        return "id: " + str(self.id) + ", fitness: " + str(self.fitness) + ", route: " + str(self.route)

    def __str__(self):
        return "id: " + str(self.id) + ", fitness: " + str(self.fitness) + ", route: " + str(self.route)


class Solver:

    def __init__(self, distanceMatrix: np.array, maxTime: int):
        n = len(distanceMatrix)                     # Amount of cities

        if n < 333:                                 # Population size
            self.lambdaa = 180
        elif 333 <= n < 666:
            self.lambdaa = 120
        elif 666 <= n:
            self.lambdaa = 60

        self.k = 3            		                # Tournament selection
        self.inversion_probability = 0.8            # Inversion Mutation probability
        self.swap_probability = 0.1                 # Swap Mutation probability
        self.mu = self.lambdaa                    	# Offspring size

        self.alpha = 0.9  # Mutation probability

        self.nn_probability = 0.01                  # Nearest Neighbors probability
        self.recombination_probability = 1          # Recombination probability

        self.mean = None                            # Mean fitness
        self.best = None                            # Best fitness
        self.bestRoute = None                       # Route of individual with best fitness

        self.iteration = 0                          # Iteration of the algorithm
        self.max_its = 40                           # Max nb of iterations
        # Maximum time for the algorithm to iterate
        self.max_time = maxTime
        self.previous_time = maxTime                # The previous iteration duration
        self.its_times = []                         # List of all iteration duration
        self.avg_its_time = 0                       # Average iteration duration
        self.previous_best = None                   # The previous best value
        # Amt of iterations in which the best value has not changed
        self.its_same_best = 0
        # The maximum amt of iterations with unchanged best value
        self.max_its_same_best = 100

        # Whether the population is split into islands
        self.ile_exploration = True
        self.ile1 = None                            # Individuals on island 1
        self.ile2 = None                            # Individuals on island 2

        # Amount of iterations before island exchange
        self.ile_its = int(round(self.max_its/10))
        # Amount of individuals swapped between iles
        self.ile_swaps = int(round(self.lambdaa/20))
        self.timeleft = 0                           # Amount of time left

        # warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

    def update_params(self):
        """
        Adapts the parameters with regard to the current state of the algorithm
        """
        self.inversion_probability = 0.8 - 0.6 * \
            (self.iteration/self.max_its)  # Inversion Mutation probability
        self.swap_probability = 0.1 + 0.3 * \
            (self.iteration/self.max_its)       # Swap Mutation probability
        self.alpha = 0.9 - 0.8*(self.iteration/self.max_its)

    def should_continue(self, time_left: int):
        """
        Returns whether the solver should continue optimizing
        """
        if (self.iteration == 0):
            # First iteration
            return True

        if (self.previous_best == None):
            self.previous_best = self.best
        if (self.previous_best == self.best):
            self.its_same_best += 1
        else:
            self.its_same_best = 0
        self.previous_best = self.best

        # Recalculate timings and iterations
        self.its_times.append(self.previous_time - time_left)
        self.avg_its_time = np.sum(self.its_times)/self.iteration
        self.previous_time = time_left
        self.max_its = self.max_time/self.avg_its_time - 10

        if (self.ile_exploration and (self.its_same_best >= self.max_its_same_best or self.iteration > 0.8*self.max_its)):
            if self.its_same_best >= int(np.ceil(self.max_its / 10)):
                self.its_same_best = 0
            self.ile_exploration = False
            return self.iteration < self.max_its and time_left > 20
        else:
            return (self.its_same_best < self.max_its_same_best
                    and self.iteration < self.max_its
                    and time_left > 20)

    def initialize(self, distanceMatrix: np.array):
        """	
        Initialize the population (random + heuristic)

        Returns the newly created population       
        """
        pop = [None for _ in range(self.lambdaa)]

        for i in range(self.lambdaa):
            ind = Individual(distanceMatrix)
            if (random.random() < self.nn_probability):
                pop[i] = nearest_neighbor(distanceMatrix, ind)
            else:
                pop[i] = ind

        self.calculateFitnesses(pop, distanceMatrix)
        return pop

    def selection(self, population: List[Individual]):
        """ 
        Selection operator 
        """
        return k_tournament(population, self.k)

    def calculateFitnesses(self, population: List[Individual], distanceMatrix: np.array):
        """ 
        Calculate the fitnesses of all the individuals in the population 
        """
        for ind in population:
            ind.computeFitness(distanceMatrix)

    def recombination(self, distanceMatrix: np.array, ind1: Individual, ind2: Individual, ileNb: int):
        """ Recombination """
        if ileNb == 1:
            c1, c2 = order_crossover(ind1, ind2)
            return [Individual(distanceMatrix, copy.deepcopy(c1)), Individual(distanceMatrix, copy.deepcopy(c2))]
        else:
            c = scx(ind1, ind2, distanceMatrix)
            return [Individual(distanceMatrix, copy.deepcopy(c))]

    def mutation(self, individual: Individual):
        """ Mutation """
        if random.random() < self.alpha:  # individual.alpha:
            r = random.random()
            if r <= self.swap_probability:
                swap_mutation(individual)
            elif r <= self.swap_probability + self.inversion_probability:
                inversion_mutation(individual)
            else:
                insert_mutation(individual)
        return individual

    def lso(self, distanceMatrix, p: Individual):
        """ Local search operator """
        p.route = list(two_opt(distanceMatrix, p.route))

    def elimination(self, population: List[Individual], offspring: List[Individual]):
        """ Elimination """
        if (self.ile_exploration):
            lambdaa = int(round(self.lambdaa/2))
        else:
            lambdaa = self.lambdaa
        return k_tournament_elimination(population, offspring, lambdaa, self.k)

    def ile_exchange(self, ile1: List[Individual], ile2: List[Individual]):
        """
        Exchange individuals between two islands.

        Returns the islands after the exchange.
        """
        for _ in range(self.ile_swaps):
            # random.randint(0, len(ile1)-1)
            index1 = k_tournament(ile1, self.k)
            i1 = ile1.index(index1)
            # random.randint(0, len(ile2)-1)
            index2 = k_tournament(ile2, self.k)
            i2 = ile2.index(index2)
            ile1[i1], ile2[i2] = ile2[i2], ile1[i1]
        return ile1, ile2

    def evolutionary_iteration(self, distanceMatrix: np.array, pop: List[Individual]):
        """ One iteration of the evolutionary algorithm """

        if (self.ile_exploration):

            if self.ile1 == None:
                self.ile1 = pop[:len(pop)//2]
                self.ile2 = pop[len(pop)//2:]

            # Iterate on island itself

            self.ile1 = self.evolutionary_iteration_ind(
                distanceMatrix, True, self.ile1, 1)
            self.ile2 = self.evolutionary_iteration_ind(
                distanceMatrix, True, self.ile2, 2)

            self.update_params()
            self.iteration += 1

            # Exchange
            if (self.iteration % self.ile_its == 0):
                self.ile1, self.ile2 = self.ile_exchange(self.ile1, self.ile2)

            pop = self.ile1 + self.ile2

        else:
            pop = self.evolutionary_iteration_ind(
                distanceMatrix, False,  pop, 1)
            self.update_params

            self.iteration += 1

        best = min(pop, key=lambda p: p.fitness)

        self.mean = np.mean([i.fitness for i in pop])
        self.best = best.fitness
        self.bestRoute = np.array(best.route)

        return pop

    def evolutionary_iteration_ind(self, distanceMatrix: np.array, isIsland, pop: List[Individual], islandNb: int):
        """
        Creates a new population
        """
        offspring = []

        if isIsland:
            its = int(np.ceil(self.mu / 4))
        else:
            its = int(np.ceil(self.mu / 2))

        for _ in range(its):
            # selection
            p1, p2 = self.selection(pop), self.selection(pop)

            # recombination
            if random.random() < self.recombination_probability:
                new_offspring = self.recombination(
                    distanceMatrix, p1, p2, islandNb)
            else:
                new_offspring = [p1, p2]

            # mutation
            new_offspring = [self.mutation(ind) for ind in new_offspring]

            offspring.extend(new_offspring)

        # lso
        for ind in offspring:
            self.lso(distanceMatrix, ind)

        self.calculateFitnesses(offspring, distanceMatrix)

        # elimination
        pop = self.elimination(pop, offspring)

        self.calculateFitnesses(pop, distanceMatrix)

        return pop


#########################################################
#                      OPERATORS                        #
#########################################################

def k_tournament(population: List[Individual], k: int):
    """
    K-Tournament operator

    Samples k random individuals from the population

    Returns the best individual from the k samples
    """
    random_k = random.sample(population, k)
    return min(random_k, key=lambda x: x.fitness)


def swap_mutation(p: Individual):
    """ 
    Mutation operator: Swap Mutation

    Takes two random indices and swaps the corresponding elements
    """
    for _ in range(p.sigma):
        i1 = random.randint(0, len(p.route)-1)
        i2 = random.randint(0, len(p.route)-1)
        p.route[i1], p.route[i2] = p.route[i2], p.route[i1]


def inversion_mutation(p: Individual):
    """ 
    Mutation operator: Inversion mutation 

    Takes two random indices and inverses the route between them
    """
    for _ in range(p.sigma):
        i1 = random.randint(0, len(p.route)-1)
        i2 = random.randint(0, len(p.route)-1)
        if i1 > i2:
            i1, i2 = i2, i1
        p.route = list(itertools.chain(p.route[:i1], list(
            reversed(p.route[i1:i2])), p.route[i2:]))


def insert_mutation(p: Individual):
    """
    Mutation Operator: Insert Mutation

    Takes two random nodes and puts the second to follow the first,
    shifting the rest along to accomodate.

    Preserves most of the order and adjacency information
    """
    for _ in range(p.sigma):
        i1 = random.randint(0, len(p.route)-1)
        i2 = random.randint(0, len(p.route)-1)
        if i1 > i2:
            i1, i2 = i2, i1
        elif i1 == i2:
            return
        p.route = list(itertools.chain(
            list(p.route[:i1+1]),
            [p.route[i2]],
            list(p.route[i1+1:i2]),
            list(p.route[i2+1:])
        ))


def basic_lso(distanceMatrix: np.array, p: Individual):
    """
    Local Search Operator: Basic

    Puts all the elements in first position and checks if fitness improves
    """
    bestFitness = p.computeFitness(distanceMatrix)
    bestInd = copy.deepcopy(p)
    copyInd = copy.deepcopy(p)

    for i in range(1, len(p.route)):
        # insert i into the first position
        copyInd.route[0] = p.route[i]
        copyInd.route[1:i+1] = p.route[0:i]
        copyInd.route[i+1:] = p.route[i+1:]
        fv = copyInd.computeFitness(distanceMatrix)

        if fv < bestFitness:
            bestFitness = fv
            bestInd.route = copyInd.route

    p.route = bestInd.route


@jit
def two_opt(distanceMatrix: np.array, route: np.array):
    """
    Local Search Operator: 2-opt for permutations

    Improve the Individuals fitness by swapping two edges in its route. 

    http://pedrohfsd.com/2017/08/09/2opt-part1.html
    """
    best = route
    improved = True
    its = 0

    while improved and its < 50:
        its += 1
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i + 2, len(route)):

                if two_opt_worth_swap(distanceMatrix, best, i, j):
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
    return best


@jit
def two_opt_worth_swap(distanceMatrix: np.array, route: np.array, i: int, j: int):
    """
    Calculates whether the possible swap on the given route on nodes with 
    index i and j lowers the total fitness
    """
    cost1 = distanceMatrix[route[i - 1]][route[j - 1]] \
        + distanceMatrix[route[i]][route[j]]

    cost2 = distanceMatrix[route[i - 1]][route[i]] \
        + distanceMatrix[route[j - 1]][route[j]]
    return cost1 < cost2


def nearest_neighbor(distanceMatrix: np.array, p: Individual):
    """
    Local Search Operator: Nearest Neighbor:
    Find the heuristic best route using nearest neighbor
    """
    start = p.route[0]
    nnroute = [start]
    unvisited = set(p.route)
    unvisited.remove(start)
    for i in range(len(p.route)-1):
        # C = nn_ind(nnroute[i], unvisited, distanceMatrix)
        C = min(unvisited, key=lambda c: distanceMatrix[nnroute[i]][c])
        nnroute.append(C)
        unvisited.remove(C)
    p.route = nnroute
    return p


def order_crossover(ind1: Individual, ind2: Individual):
    """ 
    Recombination operator: Order Crossover    

    Takes two parents and returns two offspring children
    """
    p1 = copy.deepcopy(ind1.route)
    p2 = copy.deepcopy(ind2.route)
    size = len(p1)

    a, b = random.sample(range(size), 2)

    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[p2[i]] = False
            holes2[p1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = p1, p2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            p1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            p2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        p1[i], p2[i] = p2[i], p1[i]

    return p1, p2


def scx(ind1: Individual, ind2: Individual, distanceMatrix: np.array):
    """
    Recombination operator: Sequential Constructive Crossover (SCX)

    Takes two individuals, the parents and generates a new individual, the child, using
    the scx algorithm

    Returns the child

    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.371.7771&rep=rep1&type=pdf
    """

    parent_chrom1 = ind1.route
    parent_chrom2 = ind2.route
    chrom_len = len(parent_chrom1)

    # Initialize the child
    child_chrom = [None for _ in range(len(parent_chrom1))]
    child_chrom[0] = parent_chrom1[0]
    prev_node = child_chrom[0]

    # Create a sorted set with the remaining nodes
    chrom_set = set(range(chrom_len))
    chrom_set.remove(prev_node)

    for i in range(1, len(parent_chrom1)):
        # Update the current node
        prev_node_idx1 = parent_chrom1.index(prev_node)
        prev_node_idx2 = parent_chrom2.index(prev_node)

        # Find the next legitimate node in both the parents
        if (prev_node_idx1+1 < chrom_len and parent_chrom1[prev_node_idx1+1] in chrom_set):
            leg_node_p1 = parent_chrom1[prev_node_idx1+1]
        else:
            leg_node_p1 = next(iter(chrom_set))

        if (prev_node_idx2+1 < chrom_len and parent_chrom2[prev_node_idx2+1] in chrom_set):
            leg_node_p2 = parent_chrom2[prev_node_idx2+1]
        else:
            leg_node_p2 = next(iter(chrom_set))

        # Pick the node with the lowest cost
        # if scx_cost(prev_node, leg_node_p1, distanceMatrix) > scx_cost(prev_node, leg_node_p2, distanceMatrix):
        if distanceMatrix[prev_node-1][leg_node_p1-1] > distanceMatrix[prev_node-1][leg_node_p2-1]:
            child_chrom[i] = leg_node_p2
        else:
            child_chrom[i] = leg_node_p1

        prev_node = child_chrom[i]
        chrom_set.remove(prev_node)

    return child_chrom


def lambdaplusmu_elimination(population: List[Individual], offspring: List[Individual], top_lambda) -> List[Individual]:
    """ 
    Elimination operator: lambda + mu 
    """
    return sorted(population + offspring, key=lambda p: p.fitness)[:top_lambda]


def k_tournament_elimination(population: List[Individual], offspring: List[Individual], top_lambda: int, k: int) -> List[Individual]:
    """
    Elimination operator: K-Tournament
    """
    pop = sorted(population+offspring, key=lambda k: k.fitness)
    return [pop[0]] + [k_tournament(pop, k) for _ in range(top_lambda-1)]
