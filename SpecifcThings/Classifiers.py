# Classification Algorithms TODO add a ANN
import math
import seaborn as sns
from matplotlib import pyplot as graph
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix
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


# Make.... blobs (clusters)
sample_size = 500
x_data, y_data = make_blobs(n_samples=sample_size, centers=3, random_state=1992, cluster_std=2.0)

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
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    DecisionTreeClassifier(),
    SVC(),
    SVC(gamma=1),
    AdaBoostClassifier(),
    GaussianNB(),
    BernoulliNB(),
    RandomForestClassifier(n_estimators=300, max_depth=(math.floor(math.log(sample_size/10)/math.log(2))))
]

names = [
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
    
    # Display Predictions
    plot_confusion_matrix(
        confusion_matrix(y_test, classifier.predict(x_test)),
        title=name
    )

'''
Linear Discriminate Analysis

# Train model
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

# Test Classification
lda_score = cross_val_score(lda, x_test, y_test)
print_accuracy(lda_score, 'LDA')

# Display Predictions
plot_confusion_matrix(
    confusion_matrix(y_test, lda.predict(x_test)),
    title='LDA'
)


Quadratic Discriminate Analysis

qda = QuadraticDiscriminantAnalysis()
qda.fit(x_train, y_train)

# Test Classification
qda_score = cross_val_score(qda, x_test, y_test)
print_accuracy(qda_score, 'QDA')

# Display Predictions
plot_confusion_matrix(
    confusion_matrix(y_test, qda.predict(x_test)),
    title='QDA'
)


Decision Tree Classifier

# Train model
max_tree_depth = math.floor(math.log(sample_size / 10) / math.log(2))  # This is always generate the max tree depth
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)

# Test Classification
dt_score = cross_val_score(decision_tree, x_test, y_test, cv=10)
print_accuracy(dt_score, 'DT')

# Display Predictions
plot_confusion_matrix(
    confusion_matrix(y_test, decision_tree.predict(x_test)),
    title='Decision Tree'
)


Support Vector Machine Classification

# Train SVM Model
svm_default = SVC()
svm_default.fit(x_train, y_train)

# Test Classification
svm_score = cross_val_score(svm_default, x_test, y_test, cv=10)
print_accuracy(svm_score, 'SVM')

# Display SVM Predictions
plot_confusion_matrix(
    confusion_matrix(y_test, svm_default.predict(x_test)),
    title='SVM Normal'
)


Support Vector Machine with Radial Bias Function Classification

# Train SVM RBF Model
svm_rbf = SVC(gamma=1)
svm_rbf.fit(x_train, y_train)

# Test Classification
svm_rbf_score = cross_val_score(svm_rbf, x_test, y_test, cv=10)
print_accuracy(svm_rbf_score, 'SVM RBF')

# Display SVM Predictions
plot_confusion_matrix(
    confusion_matrix(y_test, svm_rbf.predict(x_test)),
    title='SVM RBF'
)


AdaBoost Classification

# Train AdaBoost Model
adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(x_train, y_train)

# Test Classification
ab_score = cross_val_score(adaboost, x_test, y_test, cv=10)
print_accuracy(ab_score, 'Ada')

# Display AdaBoost Predictions
plot_confusion_matrix(
    confusion_matrix(y_test, adaboost.predict(x_test)),
    title='Adaboost'
)


Naive Bayes Classification Models

# Train Normal Distribution Naive Bayes Model
gaussian_bayes = GaussianNB()
gaussian_bayes.fit(x_train, y_train)

# Test Classification
g_bayes_score = cross_val_score(gaussian_bayes, x_test, y_test, cv=10)
print_accuracy(g_bayes_score, 'G-Bayes')

# Display Bayes Predictions
plot_confusion_matrix(
    confusion_matrix(y_test, gaussian_bayes.predict(x_test)),
    title='Gaussian Bayes'
)

# Train Bernoulli Bayes
bern_bayes = BernoulliNB()
bern_bayes.fit(x_train, y_train)

# Test Classification
b_bayes_score = cross_val_score(bern_bayes, x_test, y_test, cv=10)
print_accuracy(b_bayes_score, 'B-Bayes')

# Display Bayes Predictions
plot_confusion_matrix(
    confusion_matrix(y_test, bern_bayes.predict(x_test)),
    title='Bernoulli Bayes'
)


Random Forest of Decision Trees

# Train model
random_forest = RandomForestClassifier(n_estimators=300, max_depth=5)
random_forest.fit(x_train, y_train)

# Test Classification
rf_score = cross_val_score(random_forest, x_test, y_test, cv=10)
print_accuracy(rf_score, 'RF')

# Display Random Forest Predictions
plot_confusion_matrix(
    confusion_matrix(y_test, random_forest.predict(x_test)),
    title='Random Forest'
)
'''