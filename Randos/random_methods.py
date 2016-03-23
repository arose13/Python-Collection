import matplotlib.pyplot as graph
import seaborn as sns
from random import shuffle

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print('Original: ', x)

for _ in range(20):
    xi = x
    shuffle(xi)
    print('Randoms : ', xi)
    graph.plot(xi, color='gray')

graph.plot(x, color='green', linewidth=5)
graph.show()
