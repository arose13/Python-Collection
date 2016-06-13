# These are the decorator methods
def mutate_function(input_func):
    input_func.extra = 'I am extra!'
    return input_func


def generate_json(input_function):
    import json

    def inner_func(*args, **kwargs):
        do_nothing = input_function(*args, **kwargs)
        return json.dumps(do_nothing)

    return inner_func


def before_and_after(input_func):
    def inner_func(*args, **kwargs):
        print('Calling {}()'.format(input_func.__name__))
        post_func = input_func(*args, **kwargs)
        print('Got {}'.format(post_func))
        return post_func

    return inner_func

if __name__ == '__main__':
    @mutate_function
    def addition(a, b):
        return a + b

    print(addition(1, 2))
    print(addition.extra)

    @generate_json
    def jsonStuff():
        return dict(first='Anthony', last='Rose', age=float('inf'))

    print(jsonStuff())

    @before_and_after
    def array_maker(count_to: int):
        a = [x for x in range(count_to)]
        return a

    print(array_maker(5))
