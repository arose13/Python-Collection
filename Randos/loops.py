import operator


s = 'Anthony is the coolest person in the whole wide world'


def shortcut_loop_with_continue():
    for char in s:
        if char == 'e':
            continue  # This will skip everything else that the loop wanted to do! (In this case printing the character)
        print(char, end='')


def stopping_loops_with_break():
    for char in s:
        if char == 'e':
            break  # This will stop the entire loop! (In this case when it encounters the first e)
        print(char, end='')


def else_after_loops():
    for char in s:
        print(char, end='')
    else:
        print(" ELSE ")


def loop_dictionary():
    dictionary = {'x': 2, 'z': 8, 'y': 4}
    sorted_dictionary = sorted(dictionary.items(), key=operator.itemgetter(1))
    print(sorted_dictionary)

    i = 1
    for key, value in sorted_dictionary:
        if i > 2:
            break
        print(key, value)
        i += 1


if __name__ == "__main__":
    shortcut_loop_with_continue()
    print()
    stopping_loops_with_break()
    print()
    else_after_loops()
    print()
    loop_dictionary()
