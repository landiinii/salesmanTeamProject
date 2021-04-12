#!/usr/bin/python3
from BBState import BBState
from heapArray import Heap
from swap import Swap
from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' 
    <summary>
    This is the entry point for the default solver
    which just finds a valid random tour.  Note this could be used to find your
    initial BSSF.
    </summary>
    <returns>results dictionary for GUI that contains three ints: cost of solution, 
    time spent to find solution, number of permutations tried during search, the 
    solution found, and three null values for fields not used for this 
    algorithm</returns> 
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()

        while time.time() - start_time < time_allowance and count < ncities:
            citiesLeft = set(cities)
            route = [cities[count % ncities]]
            citiesLeft.remove(cities[count % ncities])
            currNext = None
            currCost = math.inf
            deadEnd = False
            # Finds the greedy route through the cities at this start point. runs in WCS n^2 time and maintains constant
            # space
            while citiesLeft:
                for city in citiesLeft:
                    thisCost = route[len(route) - 1].costTo(city)
                    if thisCost < currCost:
                        currCost = thisCost
                        currNext = city
                if currNext and currNext in citiesLeft:
                    route.append(currNext)
                    citiesLeft.remove(currNext)
                    currCost = math.inf
                else:
                    deadEnd = True
                    break
            # if a solution was found and if it was better than the previously stored solution, then the solution is
            # replaced. Constant space and time.
            if not deadEnd:
                ibssf = TSPSolution(route)
                if bssf:
                    if ibssf.cost < bssf.cost:
                        bssf = ibssf
                elif ibssf.cost < math.inf:
                    bssf = ibssf
                    foundTour = True
            count += 1

        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''
    # The Branch and bound algorithm creates first off uses a greedy algorithm to get an initial BSSf. Then it creates
    # an initial cost matrix connecting all of the cities and reduces this cost matrix to create a state on n^2 size.
    # It then loops until the priority queue of states is empty. For each state in the queue it will either
    # prune (reject) the state and all of its implicit sub states, or it will expand the state into its possible
    # substates and add each of these to the tree. This totals a run time of n^2 times the total number of states added
    # to the queue, and a space complexity of n^2 * the max number of states on the queue at any given time.
    def branchAndBound(self, time_allowance=60.0):
        # Initializes necessary variables for the algorithm to run
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        count = 0
        bssf = greedy(cities)
        start_time = time.time()

        # Initialization of Cost matrix. Computing the initial cost matrix will run in n^2 time, and take n^2 space.
        # Once it is initialize the matrix is reduced, and the initial lower bound returned. Reducing the matrix runs
        # in n^2 time, but maintains constant space.
        costMatrix = []
        for i in range(ncities):
            lengths = []
            for j in range(ncities):
                lengths.append(cities[i].costTo(cities[j]))
            costMatrix.append(lengths)
        lowerBound = rcm(costMatrix)

        # setting up the initial variables needed for the B&B algorithm includes setting up a heap. Insert into, and
        # deleting from this PQ heap are log(n) operations. The heap will store BBState objects which initialize in
        # constant time, but take up n^2 space because of the internal cost matrix they carry. The Max space of the
        # heap will be n^2 * the max number of states in the queue at one time which is reported upon completion of the
        # algorithm.
        citiesLeft = set(cities)
        route = [cities[count % ncities]]
        citiesLeft.remove(cities[count % ncities])
        problemStates = Heap()
        problemStates.insert(BBState(costMatrix, route, citiesLeft, lowerBound))
        totalStates = 1
        prunedStates = 0
        # This algorithm will run until either the time limit is exceeded, or until the Priority queue holding all of
        # the states is empty. The time running out ironically tells us nothing of the time complexity of this
        # algorithm, but the total number of states processed in the queue will tell us the time complexity
        # multiplying the total time complexity of the inner loop which is n^2 and the total number of states ever added
        # to the queue.
        while problemStates.length() > 0 and time.time() - start_time < time_allowance:
            state = problemStates.remove()
            fromCity = state.latest.getIndex()
            # For each state popped off the queue potential states are created leading from the current state to each
            # of the remaining states to be visited from the current states perspective. Expanding state costs n^2 time
            # per state that it spawns, since the state has to reduce its cost matrix. So in the WCS this over all
            # operation will cost n^3 time. This will also create a maximum of n new states which are all n^2 in size,
            # and therefore would also take up n^3 space.
            if state.lowerBound < bssf.cost:
                for s in state.remaining:
                    toCity = s.getIndex()
                    if state.matrix[fromCity][toCity] + state.lowerBound < bssf.cost:
                        stateRoute = state.currRoute[:]
                        stateRoute.append(s)
                        remainders = set(state.remaining)
                        remainders.remove(s)
                        if not remainders:
                            newSolution = TSPSolution(stateRoute)
                            if newSolution.cost < bssf.cost:
                                bssf = newSolution
                                count += 1
                        else:
                            # Construction of a new state requires calling to reduce the cost matrix which runs in n^2
                            # time. and maintains constant space.
                            newBound = state.lowerBound + state.matrix[fromCity][toCity]
                            if newBound < bssf.cost:
                                matrix = []
                                for row in state.matrix:
                                    matrix.append(row[:])
                                for i in range(len(matrix[fromCity])):
                                    matrix[fromCity][i] = math.inf
                                for row in matrix:
                                    row[toCity] = math.inf
                                matrix[toCity][fromCity] = math.inf
                                newBound += rcm(matrix)
                                if newBound < bssf.cost:
                                    totalStates += 1
                                    problemStates.insert(BBState(matrix, stateRoute, remainders, newBound))
            else:
                prunedStates += 1
        prunedStates += problemStates.length()
        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = problemStates.max
        results['total'] = totalStates
        results['pruned'] = prunedStates
        return results

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''

    def fancy(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        start_time = time.time()
        bssf = None
        count = 0
        imp = 0
        repeat = 0

        while time.time() - start_time < time_allowance and repeat < 2:
            rand = self.defaultRandomTour()['soln']
            route = rand.route
            improvement = True
            bestSwap = None

            while improvement and time.time() - start_time < time_allowance:
                improvement = False
                for i in range(len(route)):
                    bestSwap = None
                    swapFound = False
                    for j in range(len(route)):
                        if i != j:
                            if i == len(route) - 1:
                                i = - 1
                            if j == len(route) - 1:
                                j = - 1
                            changeCost = route[i - 1].costTo(route[j]) + route[j].costTo(route[i + 1]) + route[
                                j - 1].costTo(route[i]) + route[i].costTo(route[j + 1])
                            ogCost = route[i - 1].costTo(route[i]) + route[i].costTo(route[i + 1]) + route[
                                j - 1].costTo(route[j]) + route[j].costTo(route[j + 1])
                            if changeCost < ogCost:
                                if bestSwap:
                                    if bestSwap.improvement < ogCost - changeCost:
                                        bestSwap = Swap(i, j, ogCost - changeCost)
                                else:
                                    bestSwap = Swap(i, j, ogCost - changeCost)
                                swapFound = True
                    if swapFound:
                        route[bestSwap.x], route[bestSwap.y] = route[bestSwap.y], route[bestSwap.x]
                        improvement = True
            newSol = TSPSolution(route)
            count += 1
            if bssf:
                if bssf.cost == newSol.cost:
                    repeat += 1
                if bssf.cost > newSol.cost:
                    imp += 1
                    bssf = newSol
            else:
                imp += 1
                bssf = newSol
        end_time = time.time()
        results['cost'] = bssf.cost
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = imp
        results['total'] = None
        results['pruned'] = None
        return results


# The Reduce Cost Matrix function is a helper function that reduces the costMatrix it is passed, and returns the
# lower bound from that reduction. It runs in in 2 main parts. First it minimizes the rows and stores the minimum
# column values in an array. As it minimizes rows it adds to an integer tracking the current lower bound. Then in its
# second part it takes the stored column minimums and uses them to reduce the columns and sum their reductions with
# the lower bound. The whole function runs in n^2 time, and maintains constant space of what it was passed.
def rcm(matrix):
    columnMins = [math.inf] * len(matrix[0])
    lowerBound = 0
    # Minimizing the rows runs in n^2 time and maintains constant space
    for i in range(len(matrix)):
        rowMin = math.inf
        for j in range(len(matrix[i])):
            if matrix[i][j] < rowMin:
                rowMin = matrix[i][j]
            if matrix[i][j] < columnMins[j]:
                columnMins[j] = matrix[i][j]
        if math.inf > rowMin > 0:
            lowerBound += rowMin
            for j in range(len(matrix[i])):
                matrix[i][j] -= rowMin
                if matrix[i][j] < columnMins[j]:
                    columnMins[j] = matrix[i][j]
    # Minimizing columns runs in a WCS of n^2 time and a BCS of n time. Maintains Constant Space
    for k in range(len(columnMins)):
        if 0 < columnMins[k] < math.inf:
            lowerBound += columnMins[k]
            for i in range(len(matrix)):
                matrix[i][k] -= columnMins[k]
    return lowerBound


# This greedy algorithm runs through a conventional greedy algorithm n times, one for a starting point at each potential
# city. It implements a greedy that will run in WCS Summation(n) time, which we will approximate to be n^2. Which means
# that the function has an overall BidO(n^3) and a space complexity of n.
def greedy(cities):
    ncities = len(cities)
    bssf = None
    # runs a greedy algorithm at every possible start point (n) in the set of cities
    for i in range(ncities):
        citiesLeft = set(cities)
        route = [cities[i % ncities]]
        citiesLeft.remove(cities[i % ncities])
        currNext = None
        currCost = math.inf
        deadEnd = False
        # Finds the greedy route through the cities at this start point. runs in WCS n^2 time and maintains constant
        # space
        while citiesLeft:
            for city in citiesLeft:
                thisCost = route[len(route) - 1].costTo(city)
                if thisCost < currCost:
                    currCost = thisCost
                    currNext = city
            if currNext and currNext in citiesLeft:
                route.append(currNext)
                citiesLeft.remove(currNext)
                currCost = math.inf
            else:
                deadEnd = True
                break
        # if a solution was found and if it was better than the previously stored solution, then the solution is
        # replaced. Constant space and time.
        if not deadEnd:
            ibssf = TSPSolution(route)
            if bssf:
                if ibssf.cost < bssf.cost:
                    bssf = ibssf
            elif ibssf.cost < math.inf:
                bssf = ibssf

    print(str(bssf.cost))
    return bssf
