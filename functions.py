def arbitrary_args(required, *args):
    # The *args contains a tuple of arbitrary number of arguments
    print('--------------------------------')
    print(required, args)

    if len(args) > 0:
        print(args[0])

    for n in args:  # Yes! You can loop through each argument!
        print(n)


def dictionary_args(**kwargs):
    # **kwargs is actually a 'dictionary' of args. This might be useful :/
    print(type(kwargs))
    print(kwargs['one'], kwargs['two'])


def fancy_args(required, *args, **kwargs):
    print('Holy Shit', required, args, kwargs['this'], kwargs['that'])


if __name__ == '__main__':
    arbitrary_args(1)
    arbitrary_args(1, 2, 3, 4, 5)
    arbitrary_args(6, 7, 8)
    print('----------------------------')
    dictionary_args(one=1, two=2)
    print('----------------------------')
    fancy_args('Ants', 1, 2, 3, that="Scheisser", this="Heilger")
