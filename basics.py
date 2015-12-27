def main():
    person = "Anthony"
    print(person + "'s Awesome")


def do_not_run():
    for i in range(10):
        print("This shouldn't run")


def function(a="blah", b=2):
    # This is a python style function
    print(a, b)


class Egg:
    # This is a python constructor method
    # Everything after self are
    # The constructor is called everytime you call a
    def __init__(self, kind="fried"):
        self.kind = kind

    def get_kind(self):
        return self.kind


def make_breakfast():
    fried_eggs = Egg()
    scrambled_eggs = Egg(kind="scrambled")
    print(fried_eggs.get_kind())
    print(scrambled_eggs.get_kind())

# If this class has been called itself and not just simply included run this section
if __name__ == "__main__":
    main()  # This can be any function that you want to run
    function()
    function(a="Replaced A")
    function(b=422)
    function("Java Style", 123)
    function(a="Python Style", b=456)
    make_breakfast()
