# Perceptron from scratch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plot
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Get data
iris_data = load_iris()
x_data = iris_data.data[:, [2, 3]]
y_data = iris_data.target

# Standardise X
standardise = StandardScaler()
standardise.fit_transform(x_data)

print('Number of classes (Expect 3)')
print(np.unique(y_data))

# Split Data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1992)

# Initialise Learning Model
ptron = Perceptron(n_iter=100, eta0=0.1, random_state=1992)
ptron.fit(x_train, y_train)

y_predicted = ptron.predict(x_test)
print('Mis-classified sample: {}'.format((y_test != y_predicted).sum()))
print('Accuracy: {}'.format(accuracy_score(y_test, y_predicted)))

# Display Information
sns.set()


def plot_decision_regions(resolution=0.02):
    # Plot decision surface
    x1_min, x1_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    x2_min, x2_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = ptron.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plot.contourf(xx1, xx2, z, alpha=0.25, cmap='rainbow')

    # Plot Data Points
    plot.scatter(x_test[:, 0], x_test[:, 1], marker='o', c=y_predicted, cmap='rainbow', label='Test')
    plot.scatter(x_train[:, 0], x_train[:, 1], marker='v', c=y_train, cmap='rainbow', label='Train')
    plot.legend(loc=0)
    plot.show()

plot_decision_regions()

