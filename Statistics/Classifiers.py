# Classification Algorithms TODO add a ANN
import math
import numpy as np
import seaborn as sns
from matplotlib import pyplot as graph
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.learning_curve import learning_curve
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

__author__ = 'Anthony Rose'
sns.set()


# Accuracy Metrics
def print_accuracy(input_score, title):
    print('{} Accuracy: {} +/- {} std'.format(title, input_score.mean(), input_score.std() * 2))


def plot_confusion_matrix(cm, title='Model', cmap=graph.cm.Greens):
    graph.imshow(cm, interpolation='nearest', cmap=cmap)
    graph.title(title + ' Confusion Matrix')
    graph.colorbar()
    graph.axis('tight')
    graph.ylabel('True label')
    graph.xlabel('Predicted label')
    graph.show()


def plot_decision_regions(x, xtrain, xtest, ytrain, clf, title='X', resolution=0.02):
    # Plot decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    graph.contourf(xx1, xx2, z, alpha=0.25, cmap='rainbow')
    graph.xlim(xx1.min(), xx1.max())
    graph.ylim(xx2.min(), xx2.max())

    # Plot Data Points
    graph.scatter(xtest[:, 0], xtest[:, 1], marker='x', c=clf.predict(xtest), cmap='rainbow', label='Predicted')
    graph.scatter(xtrain[:, 0], xtrain[:, 1], marker='o', c=ytrain, cmap='rainbow', label='Train')
    graph.legend(loc=0)
    graph.title(title)
    graph.show()


def plot_learning_curve(xtrain, ytrain, clf):
    train_sizes, train_scores, test_scores = learning_curve(
            clf, xtrain, ytrain, train_sizes=np.linspace(0.1, 1.0, 10), cv=10
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Display Learning Graph
    # Training Curve
    graph.plot(train_sizes, train_mean, color='blue', label='Training Acc')
    graph.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    # Testing Curve
    graph.plot(train_sizes, test_mean, color='green', label='Validating Acc')
    graph.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    graph.grid()
    graph.xlabel('Training Sample Size')
    graph.ylabel('Accuracy (p)')
    graph.legend(loc=0)
    graph.show()


def plot_roc_auc_stats(y_target, y_predicted, title='Model'):
    # Receiver Operator Characteristic Area Under the Curve
    # FPR = False Positive Rate, TPR = True Positive Rate
    graph_limits = [-0.1, 1.1]
    fpr, tpr, thresholds = roc_curve(y_target, y_predicted)

    graph.plot(fpr, tpr, label='{} (AUC = {})'.format(title, auc(x=fpr, y=tpr)))
    graph.legend(loc='lower right')
    graph.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
    graph.xlim(graph_limits)
    graph.ylim(graph_limits)
    graph.xlabel('False Positive Rates')
    graph.ylabel('True Positive Rates')
    graph.show()


# Make.... blobs (clusters)
sample_size = 500
x_data, y_data = make_blobs(n_samples=sample_size, centers=2, random_state=1992, cluster_std=2.0)

graph.title('All Data')
graph.scatter(x_data[:, 0], x_data[:, 1], s=50, c=y_data, cmap='rainbow')
graph.show()

# Standardise Data
x_data = StandardScaler().fit_transform(x_data)  # Standard when the units are different

graph.title('Standardised Data')
graph.scatter(x_data[:, 0], x_data[:, 1], s=50, c=y_data, cmap='rainbow')
graph.show()

# Training and testing data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
graph.title('Training Data')
graph.scatter(x_train[:, 0], x_train[:, 1], s=50, c=y_train, cmap='rainbow')
graph.show()

'''
Loop Through Classifiers
'''

classifiers = [
    Perceptron(n_iter=100, eta0=0.1, random_state=1992),
    LogisticRegressionCV(Cs=50, max_iter=500),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    SVC(),
    SVC(gamma=1),
    AdaBoostClassifier(),
    GaussianNB(),
    BernoulliNB(),
    RandomForestClassifier(n_estimators=300, max_depth=(math.floor(math.log(sample_size / 10) / math.log(2))))
]

names = [
    'Ptron',
    'Logit',
    'LDA',
    'QDA',
    'DT',
    'SVC',
    'SVC RBF',
    'AdaB',
    'G-NB',
    'B-NB',
    'RF'
]

for name, classifier in zip(names, classifiers):
    # Train
    classifier.fit(x_train, y_train)

    # Classification
    score = cross_val_score(classifier, x_test, y_test)
    print_accuracy(score, name)

    # Predict
    y_predicted = classifier.predict(x_test)

    # Display Performance
    plot_learning_curve(x_train, y_train, classifier)

    plot_confusion_matrix(
            confusion_matrix(y_test, y_predicted),
            title=name
    )

    plot_decision_regions(
            x_data,
            x_train,
            x_test,
            y_train,
            classifier,
            title=name
    )
