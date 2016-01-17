a = 42
b = 9


def division():
    # Returns a float
    print(a / b)


def floor():
    # This will return an int
    print(a // b)


def rounding():
    print(round(a / b))
    print(round(a / b, 1))
    print(round(a / b, 2))
    print(round(a / b, 5))


def remainder():
    print(42 % 9)


def fancy_string():
    # Put number in strings
    str = "This number {} is so cool".format(a)
    print(str)


def lines_of_text():
    str = '''\
    These are loads
    of fucking lines
    of text bros!!!
    '''
    print(str)

if __name__ == "__main__":
    # Run whatever you want
    division()
    floor()
    rounding()
    fancy_string()
    lines_of_text()
    