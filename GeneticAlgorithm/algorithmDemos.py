# Trying to solve this problem in a smaller number of generations and more reliably
from scipy.optimize import differential_evolution
from GeneticAlgorithm.costFunctions import complex_bowl, rosenbrock_function, string_match_score_function, generate_string_from_array
from GeneticAlgorithm.ParticleSwarmOptimiser import swarm

if __name__ == '__main__':
    title = '\n---------------\n{}\n---------------\n'

    lower_bounds = [-20, -20]
    upper_bounds = [20, 20]
    target_string = 'Hello World'

    # Anthony
    print(title.format('Custom'))

    print('Complex Bowl')
    optimal_x, optimal_value = swarm(complex_bowl, lb=lower_bounds, ub=upper_bounds)
    print(optimal_x, optimal_value, '\n')

    print('Rosenbrock Function')
    optimal_x, optimal_value = swarm(rosenbrock_function, lb=lower_bounds, ub=upper_bounds)
    print(optimal_x, optimal_value, '\n')

    # Scipy
    print(title.format('Differential Evolution'))

    print('Complex Bowl')
    bowl = differential_evolution(complex_bowl, [(-20, 20)] * 2, strategy='best1bin')
    print(bowl, '\n')

    print('Rosenbrock Function')
    rosen = differential_evolution(rosenbrock_function, [(-20, 20)] * 2, strategy='best1bin')
    print(rosen, '\n')

    print('String Evolution')
    string_evolution = differential_evolution(
        string_match_score_function, strategy='best1bin',
        bounds=[(32, 126)] * len(target_string), args=(target_string, True))
    print(string_evolution)
    print(generate_string_from_array(string_evolution.x.tolist()))
