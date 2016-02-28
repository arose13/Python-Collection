# Playing with numpy
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
b = np.array([5, 4, 9, 5, 4, 9])

c = b / a

print('input array', c)  # Expected [5, 2, 3, 1.25, 0.8, 1.5]

# Select only the items that fulfil a requirement
d = c[np.where(c >= 2)]
print('where greater than 2', d)

e = c[np.where(c >= np.mean(c))]
print('where greater than average', e)
