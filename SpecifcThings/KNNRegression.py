# kNN Regression. Maybe cheating maybe not :D. kNN Regressions are sensitive to outliers
import numpy as np
from matplotlib import pyplot as graph
from matplotlib import pylab
from sklearn import neighbors

np.random.seed(1992)
x_range = 10
x = np.sort(x_range * np.random.rand(100, 1), axis=0)  # Produce a random list of x values
t = np.linspace(0, x_range, 500)[:, np.newaxis]  # list of x values to be predicted
y = np.sin(x).ravel()  # The function to be approximated

# Adding noise to data
noise_magnitude = 1  # kNN is sensitive to outliers
y[::x_range] += noise_magnitude * (0.5 - np.random.rand(x_range))

# Run kNN Regression
knn_weight = 'uniform'
k3 = 3
k5 = 5
k8 = 8

# Make knn models
knn3 = neighbors.KNeighborsRegressor(n_neighbors=k3, weights=knn_weight)
knn5 = neighbors.KNeighborsRegressor(n_neighbors=k5, weights=knn_weight)
knn8 = neighbors.KNeighborsRegressor(n_neighbors=k8, weights=knn_weight)

# Train Models
y_est3 = knn3.fit(x, y).predict(t)
y_est5 = knn5.fit(x, y).predict(t)
y_est8 = knn8.fit(x, y).predict(t)
y_knn_average = (y_est3 + y_est5 + y_est8) / 3

# Graph Results
graph.scatter(x, y, color='k', label='input data')
graph.plot(t, y_est3, label='k3', alpha=0.3)
graph.plot(t, y_est5, label='k5', alpha=0.3)
graph.plot(t, y_est8, label='k8', alpha=0.3)
graph.plot(t, y_knn_average, color='b', lw=2, label='Average Prediction')
graph.grid()
graph.legend()
graph.title('Combine kNN Results')
graph.show()


########################################################################################################################
# Now let's try multidimensional kNN Regressions
def f(x, y):
    # This is the function to be approximated
    return (x ** 2) * (y ** 2)


def sse(actual, predicted):
    return np.sum((actual - predicted) ** 2) / len(actual)


# Set random value so repeatability
np.random.seed(1992)

x_data = np.linspace(-3, 3, num=100)
y_data = np.linspace(-3, 3, num=100)
training_points = []

# Solve the z values for the grid of values
xx, yy = pylab.meshgrid(x_data, y_data)
zz = pylab.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[0]):
        # Combine x and y data to [x, y]
        training_points.append([xx[i, j], yy[i, j]])
        zz[i, j] = f(xx[i, j], yy[i, j]) + (10 * (np.random.random() - 0.5))  # This is a noise factor

pylab.pcolor(xx, yy, zz)
pylab.colorbar()
# Show the graph after everything has ran

# Run kNN Regression
k5 = 5
k8 = 8
k13 = 13

# Generate Models
knn5 = neighbors.KNeighborsRegressor(n_neighbors=k5, weights='uniform')
knn8 = neighbors.KNeighborsRegressor(n_neighbors=k8, weights='uniform')
knn13 = neighbors.KNeighborsRegressor(n_neighbors=k13, weights='uniform')

# NOTE there is a matrix math way of doing this! I just can't remember
training_targets = []
for pos in range(len(training_points)):
    training_targets.append(f(training_points[pos][0], training_points[pos][1]))

# Train kNN models
trained_k5_model = knn5.fit(training_points, training_targets)
trained_k8_model = knn8.fit(training_points, training_targets)
trained_k13_model = knn13.fit(training_points, training_targets)

# Test Predictions
number_of_test_points = 100
random_test_x = np.random.rand(number_of_test_points, 1) * 6 - 3
random_test_y = np.random.rand(number_of_test_points, 1) * 6 - 3
test_points = np.column_stack((random_test_x, random_test_y))

actual_values = f(random_test_x, random_test_y).T
predicted_k5 = trained_k5_model.predict(test_points)
predicted_k8 = trained_k8_model.predict(test_points)
predicted_k13 = trained_k13_model.predict(test_points)
predicted_average = (predicted_k5 + predicted_k8 + predicted_k13) / 3

print('PREDICTIONS\n---------------------------------------------------')
print('True function values')
print(actual_values)
print('k = {}'.format(k5))
print(predicted_k5)
print('k = {}'.format(k8))
print(predicted_k8)
print('k = {}'.format(k13))
print(predicted_k13)

# Test how much better the average predictions are
print('k Average')
print(predicted_average)

print('ERRORS\n---------------------------------------------------')

error_k5 = sse(actual_values, predicted_k5)
error_k8 = sse(actual_values, predicted_k8)
error_k13 = sse(actual_values, predicted_k13)
error_average = sse(actual_values, predicted_average)

print('k = {} error {}'.format(k5, error_k5))
# print(actual_values - predicted_k5)
print('k = {} error {}'.format(k8, error_k8))
# print(actual_values - predicted_k8)
print('k = {} error {}'.format(k13, error_k13))
# print(actual_values - predicted_k13)
print('k Average error {}'.format(error_average))
# print(actual_values - predicted_average)

# Display Graph Topology
pylab.show()

# CONCLUSION? The average of a few kNN regressions do a better job at approximating the function's TRUE value.
# This is evident by the decreased error even when the inputs are noisy.
