import math


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


def string_match_score_function(letter_array, *args):
    """
    Big is worse scoring function for judging the quality of a string match.
    """
    # Setup
    verbose = args[1]
    target = args[0]
    fitness = 0

    # Scoring Section
    for i, letter in enumerate(letter_array):
        fitness += abs(letter - ord(target[i]))

    # Debug
    test_string = generate_string_from_array(letter_array)
    if verbose:
        print(test_string)

    # Check solution to accelerate termination
    if test_string == target:
        fitness = 0

    return fitness ** math.pi


def generate_string_from_array(float_array):
    letter_list = [chr(int(round(x))) for x in float_array]
    string = ''.join(letter_list)
    # print(string)
    return string
