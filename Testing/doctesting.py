# Doc Testing
import doctest


def square(x):
    """
    Simplest test ever!
    >>> square(3)
    9
    >>> square(5)
    25
    >>> square(-2)
    4

    :param x:
    :return:
    """
    return x ** 2

if __name__ == '__main__':
    doctest.testmod()
