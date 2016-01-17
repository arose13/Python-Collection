def do_stuff():
    for i in range(5):
        # Notice that this range is exclusive to 5
        print(i, end=' ')

    print("\n")

    for i in inclusive_range(5):
        # This is a custom generator function that will include the last number
        print(i, end=' ')


def inclusive_range(*args):
    # Default values
    start = 0
    step = 1

    numargs = len(args)
    if numargs == 0:
        raise TypeError('Minimum stop argument required')
    elif numargs == 1:
        stop = args[0]
    elif numargs == 2:
        (start, stop) = args
    elif numargs == 3:
        (start, stop, step) = args
    else:
        raise TypeError('Unexpected number of arguments inserted')

    i = start
    while i <= stop:
        yield i  # This is a special kind of return that is NON TERMINATING
        i += step


if __name__ == '__main__':
    do_stuff()
