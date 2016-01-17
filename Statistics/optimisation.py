# Optimisation. Finding minimums
import numpy as np
import seaborn as sns
from matplotlib import pyplot as graph
from scipy import optimize as spo

__author__ = 'Anthony Rose'
sns.set()


def f(x, y):
    return 0.05 * x ** 2 + 0.05 * y ** 2 + np.sin(x) + np.sin(y)


def f_bf(params):
    z = f(params[0], params[1])
    brute_force_progress.append(z)
    return z


def f_lo(params):
    z = f(params[0], params[1])
    local_optimised_progress.append(z)
    return z


# Generate function space (not required for optimisation, just doing it for visualisation)
x_data = np.linspace(-10, 10, 100)
y_data = np.linspace(-10, 10, 100)

x_grid, y_grid = np.meshgrid(x_data, y_data)
z_grid = f(x_grid, y_grid)

# Brute Force Optimisation
brute_force_progress = []
print('-------------------------------------')
print('Brute Force Optimisation')

bf_optimised_vars = spo.brute(f_bf, ((-10, 10.1, 1), (-10, 10.1, 1)), disp=True)
print('Best Values Found is: {}'.format(bf_optimised_vars))

graph.title('Brute Force')
graph.plot(brute_force_progress)
graph.show()

# Local Optimisation
local_optimised_progress = []
print('-------------------------------------')
print('Local Optimisation from Brute Force')

local_optimised_vars = spo.fmin(f_lo, bf_optimised_vars, xtol=1e-5, ftol=1e-5, maxiter=2e3, maxfun=3e3, disp=True)
print('Bettered Brute Force Result: {}'.format(local_optimised_vars))

graph.title('Local Optimise of Brute Force')
graph.plot(local_optimised_progress)
graph.show()
