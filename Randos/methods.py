class Car:  # Notice that making the constructor is optional
    def vroom(self):
        print('brmm')

    def drive(self):
        print('Driving')


class SuperCar(Car):  # This is how inheritance works in python!
    def __init__(self, weight=1600):
        self._weight = weight

    def vroom(self):
        print('VROOOMMMM')

    def get_weight(self):
        print("Weighs: {}".format(self._weight))


def main():
    audi = Car()
    audi.vroom()
    audi.vroom()
    audi.drive()
    print()
    ferrari = SuperCar(weight=1420)
    ferrari.vroom()
    ferrari.vroom()
    ferrari.drive()
    ferrari.get_weight()


if __name__ == '__main__':
    main()
