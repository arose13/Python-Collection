# Trying to solve this problem in a smaller number of generations and more reliably
import math
from GeneticAlgorithm.ParticleSwarmOptimiser import swarm


def complex_bowl(x):
    """
    Function: z = (sin(x*pi)+0.5*x^2)+(sin(y*pi)+0.5*y^2)
    Expected Minimum = -1.77304
    Expected X @ min = (-0.453854, -0.453854)
    """
    assert len(x) == 2

    x, y = x[0], x[1]
    z = (math.sin(x * math.pi) + 0.5 * x ** 2) + (math.sin(y * math.pi) + 0.5 * y ** 2)
    return z


def rosenbrock_function(x):
    """
    Function: z = [(a-x)^2]+[b*(y-x^2)^2]
    Expected Minimum = 0.0
    Expected X @ min = (a, a^2)
    """
    assert len(x) == 2

    x, y = x[0], x[1]
    a = 1
    b = 100
    z = ((a - x) ** 2) + (b * (y - x ** 2) ** 2)
    return z


if __name__ == '__main__':
    lower_bounds = [-20, -20]
    upper_bounds = [20, 20]

    print('Complex Bowl')
    optimal_x, optimal_value = swarm(complex_bowl, lb=lower_bounds, ub=upper_bounds)
    print(optimal_x, optimal_value, '\n')

    print('Rosenbrock Function')
    optimal_x, optimal_value = swarm(rosenbrock_function, lb=lower_bounds, ub=upper_bounds)
    print(optimal_x, optimal_value)
