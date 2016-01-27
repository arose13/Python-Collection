from itertools import combinations
import numpy as np
import seaborn as sns
import matplotlib.pyplot as graph
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

    def best_features(self):
        """
        Returns the minimum number of features with the highest score.
        :return:
        """
        best_subset = None
        best_score = 0

        # TODO There must be a more python way of doing this
        for score, subset in zip(self.scores_, self.subsets_):
            if best_subset is None:
                best_subset = subset

            if score >= best_score and len(subset) < len(best_subset):
                best_score = score
                best_subset = subset

        return best_subset

    def transform(self, x):
        """
        This will best feature found via back selection
        """
        return x[:, self.best_features()]

    def calc_score(self, x_train, y_train, x_test, y_test, indices):
        self.model.fit(x_train[:, indices], y_train)
        y_predicted = self.model.predict(x_test[:, indices])
        score = self.scoring(y_test, y_predicted)
        return score

    def graph_performance(self):
        k_feature = [len(k) for k in self.subsets_]
        # Graph Accuracy vs Number of features
        graph.plot(k_feature, self.scores_, marker='o')
        graph.ylabel('Accuracy')
        graph.xlabel('Number of Features')
        graph.show()


if __name__ == '__main__':
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import train_test_split

    # Load
    sns.set()
    data = load_breast_cancer()
    x = StandardScaler().fit_transform(data.data)
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # Run Feature Selection
    model = KNeighborsClassifier(n_neighbors=2)
    sbfs = SBFS(model, k_features=1)
    sbfs.fit(
        x_train,
        y_train
    )

    # Observe Results
    best_features = sbfs.best_features()
    x_best_train, x_best_test = sbfs.transform(x_train), sbfs.transform(x_test)

    print(best_features)
    print('X', x_test.shape)
    print('X Best', x_best_test.shape)

    # Compare Best Features vs All Features Results
    # ALL model
    model.fit(x_train, y_train)
    print('\nALL MODEL')
    print('Training Accuracy {}'.format(model.score(x_train, y_train)))
    print('Test Accuracy {}'.format(model.score(x_test, y_test)))

    # BEST FEATURES model
    model.fit(x_best_train, y_train)
    print('\nBEST MODEL')
    print('Training Accuracy {}'.format(model.score(x_best_train, y_train)))
    print('Test Accuracy {}'.format(model.score(x_best_test, y_test)))
