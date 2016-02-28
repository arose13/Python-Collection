# ROC Curve for Classifiers on the breast cancer dataset
# By using One vs Rest this allows the ROC curve to generalise to an unlimited number of classes.
import matplotlib.pyplot as graph
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc


# Graphing setup
sns.set(font_scale=1.25)

# Import Data
boobs = load_breast_cancer()
x, y = boobs.data, boobs.target

# Convert classes to binaries for the output
y = label_binarize(y, classes=[0, 1])  # Only 2 because the cancer can either be there or not.
num_classes = y.shape[1]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, random_state=1992)

# Train model
model = OneVsRestClassifier(LogisticRegressionCV())
model.fit(x_train, y_train)

y_score = model.decision_function(x_test)

# Compute the ROC curve and ROC area for each class
false_positive_rate, true_positive_rate, roc_auc = dict(), dict(), dict()

for i in range(num_classes):
    false_positive_rate[i], true_positive_rate[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

# Compute average ROC curve and ROC area
false_positive_rate['micro'], true_positive_rate['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc['micro'] = auc(false_positive_rate['micro'], true_positive_rate['micro'])

# Plot ROC curve for a specific class
for i in range(num_classes):
    graph.plot(
        false_positive_rate[i],
        true_positive_rate[i],
        label='ROC curve #{} (area = {}) N={}'.format(
            i+1,
            roc_auc[i],
            len(y_test)
        ))
graph.plot([0, 1], [0, 1], 'k--')
graph.xlim(0.0, 1.0)
graph.ylim(0.0, 1.0)
graph.xlabel('False Positive Rate')
graph.ylabel('True Positive Rate')
graph.title('ROC')
graph.legend(loc=0)
graph.show()


