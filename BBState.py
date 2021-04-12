class BBState:
    def __init__(self, matrix, route, remaining, lowerBound):
        self.matrix = matrix  # the cost matrix of the current state
        self.currRoute = route  # the route leading up to this state
        self.remaining = remaining  # the set of remaining cities to visit
        self.lowerBound = lowerBound  # the updated lower bound of the state
        self.level = len(self.currRoute) - 1  # the depth of the state in the tree
        self.latest = self.currRoute[self.level]  # the last city visited
