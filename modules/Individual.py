
class Individual:
    """
    Represents one candidate solution to the optimization problem.
    This class encapsulates the parameters of the individual and its fitness value.
    for your tuning/optimization routines.
    """
    def __init__(self, params: dict, fitness: float = None):
        self.params = params       
        self.fitness = fitness     

    def __lt__(self, other):
        # allows sorting individuals by fitness
        return self.fitness < other.fitness
