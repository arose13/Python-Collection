# Assess feature importance using a random forest classifier
import numpy as np
import seaborn as sns
from matplotlib import pyplot as graph
from sklearn.ensemble import RandomForestClassifier

__author__ = 'Anthony Rose'


class EstimateFeatureImportance:
    """
    This uses a random forest to assess each feature's importance
    """
    def __init__(self, x, y_class, x_labels):
        self.x = x
        self.y_class = y_class
        self.x_labels = x_labels

        rf = RandomForestClassifier(n_estimators=10000, random_state=1992)
        rf.fit(x, y_class)

        self.importance = rf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        for feature in range(x.shape[1]):
            print('{}) {} = {}%'.format(
                    feature + 1,
                    x_labels[feature],
                    round(self.importance[self.indices[feature]], 3)*100
            ))

    def graph_importance(self):
        graph.title('Feature Importance')
        graph.bar(
            range(self.x.shape[1]),
            self.importance[self.indices],
            align='center'
        )
        graph.xticks(
            range(self.x.shape[1]),
            self.x_labels,
            rotation=90
        )
        graph.xlim([-1, self.x.shape[1]])
        graph.show()


if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_breast_cancer
    sns.set()

    # Test with the iris dataset
    iris = load_iris()
    x_iris = iris.data
    y_iris = iris.target

    # Run Feature Importance
    feature_importance = EstimateFeatureImportance(x_iris, y_iris, iris.feature_names)
    feature_importance.graph_importance()
