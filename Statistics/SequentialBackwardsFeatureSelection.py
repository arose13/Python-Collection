from itertools import combinations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as graph
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

__author__ = 'Anthony Rose'


class SBFS:
    """
    Sequential Backwards Feature Selection

    Scikit-Learn style class so it blends in with everything else better
    """

    def __init__(self, model, k_features, scoring=accuracy_score, test_size=0.2, random_state=1992):
        self.scoring = scoring
        self.model = model
        self.k_features = k_features
        self.test_size = test_size
        self.random_states = random_state

        self.indices_, self.subsets_, self.scores_, self.k_score_ = None, None, None, None

    def fit(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=self.test_size,
                random_state=self.random_states
        )

        dimensions = x_train.shape[1]
        self.indices_ = tuple(range(dimensions))
        self.subsets_ = [self.indices_]
        score = self.calc_score(x_train, y_train, x_test, y_test, self.indices_)

        self.scores_ = [score]

        while dimensions > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=(dimensions - 1)):
                score = self.calc_score(x_train, y_train, x_test, y_test, p)

                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dimensions -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, x):
        """
        This will best feature found via back selection
        """
        return x[:, self.indices_]

    def calc_score(self, x_train, y_train, x_test, y_test, indices):
        self.model.fit(x_train[:, indices], y_train)
        y_predicted = self.model.predict(x_test[:, indices])
        score = self.scoring(y_test, y_predicted)
        return score

    def graph_performance(self):
        k_feature = [len(k) for k in self.subsets_]
        # Graph Accuracy vs Number of features
        graph.plot(k_feature, self.scores_, marker='o')
        graph.title('Accuracy vs Features Used')
        graph.ylabel('Accuracy')
        graph.xlabel('Number of Features')
        graph.show()


if __name__ == '__main__':
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler

    # Load
    sns.set()
    data = load_breast_cancer()

    # Run Feature Selection
    classifier = KNeighborsClassifier(n_neighbors=2)
    sbfs = SBFS(classifier, k_features=1)
    sbfs.fit(
        StandardScaler().fit_transform(data.data),
        data.target
    )
    sbfs.graph_performance()
