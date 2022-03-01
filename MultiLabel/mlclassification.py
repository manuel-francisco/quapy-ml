from copy import deepcopy

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from skmultilearn.adapt import MLTSVM

from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.cluster.networkx import NetworkXLabelGraphClusterer
from skmultilearn.cluster.base import LabelCooccurrenceGraphBuilder

# from skmultilearn.embedding import SKLearnEmbedder, EmbeddingClassifier
from sklearn.manifold import SpectralEmbedding
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN


class MLStackedClassifier:  # aka Funnelling Monolingual
    def __init__(self, base_estimator=LogisticRegression()):
        if not hasattr(base_estimator, 'predict_proba'):
            print('the estimator does not seem to be probabilistic: calibrating')
            base_estimator = CalibratedClassifierCV(base_estimator)
        self.base = deepcopy(OneVsRestClassifier(base_estimator))
        self.meta = deepcopy(OneVsRestClassifier(base_estimator))
        self.norm = StandardScaler()

    def fit(self, X, y):
        assert y.ndim==2, 'the dataset does not seem to be multi-label'
        self.base.fit(X, y)
        P = self.base.predict_proba(X)
        P = self.norm.fit_transform(P)
        self.meta.fit(P, y)
        return self

    def predict(self, X):
        P = self.base.predict_proba(X)
        P = self.norm.transform(P)
        return self.meta.predict(P)

    def predict_proba(self, X):
        P = self.base.predict_proba(X)
        P = self.norm.transform(P)
        return self.meta.predict_proba(P)


class MLStackedRegressor:
    def __init__(self, base_regressor=Ridge(normalize=True)):
        self.base = deepcopy(base_regressor)
        self.meta = deepcopy(base_regressor)

    def fit(self, X, y):
        assert y.ndim==2, 'the dataset does not seem to be multi-label'
        self.base.fit(X, y)
        R = self.base.predict(X)
        # R = self.norm.fit_transform(R)
        self.meta.fit(R, y)
        return self

    def predict(self, X):
        R = self.base.predict(X)
        # R = self.norm.transform(R)
        return self.meta.predict(R)


class LabelSpacePartion:
    def __init__(self, base_estimator=LogisticRegression()):
        graph_builder = LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False)
        self.classifier = LabelSpacePartitioningClassifier(
            classifier=LabelPowerset(classifier=base_estimator),
            clusterer=NetworkXLabelGraphClusterer(graph_builder, method='louvain')
        )

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X).todense().getA()


class MLTwinSVM:
    def __init__(self):
        self.classifier = MLTSVM()

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X).todense().getA()


class MLknn:
    #http://scikit.ml/api/skmultilearn.embedding.classifier.html#skmultilearn.embedding.EmbeddingClassifier
    #notes: need to install package openne
    def __init__(self):
        self.classifier = EmbeddingClassifier(
            SKLearnEmbedder(SpectralEmbedding(n_components=10)),
            RandomForestRegressor(n_estimators=10),
            MLkNN(k=5)
        )

    def fit(self, X, y):
        return self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X).todense().getA()

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

