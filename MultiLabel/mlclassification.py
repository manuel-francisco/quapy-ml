from copy import deepcopy

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from skmultilearn.adapt import MLTSVM

from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.cluster.networkx import NetworkXLabelGraphClusterer
from skmultilearn.cluster.base import LabelCooccurrenceGraphBuilder
from skmultilearn.embedding import CLEMS, EmbeddingClassifier

# from skmultilearn.embedding import SKLearnEmbedder, EmbeddingClassifier
from sklearn.manifold import SpectralEmbedding
from sklearn.ensemble import RandomForestRegressor
from skmultilearn.adapt import MLkNN
from skmultilearn.cluster import MatrixLabelSpaceClusterer

from scipy.sparse import issparse
import numpy as np

import quapy as qp
from MultiLabel.mldata import MultilabelledCollection

class SKMLWrapper:
    def __init__(self, skmlclf):
        self.model_ = skmlclf

    @staticmethod
    def _todense(A):
        aux = A
        if issparse(aux):
            aux = aux.todense()
        return np.array(aux)
    
    def fit(self, X, y, *args, **kwargs):
        return self.model_.fit(np.matrix(SKMLWrapper._todense(X)), np.matrix(SKMLWrapper._todense(y)), *args, **kwargs)
    
    def predict(self, X, *args, **kwargs):
        return SKMLWrapper._todense(self.model_.predict(SKMLWrapper._todense(X), *args, **kwargs))
    
    def predict_proba(self, X, *args, **kwargs):
        return SKMLWrapper._todense(self.model_.predict_proba(np.matrix(SKMLWrapper._todense(X)), *args, **kwargs))
        probas = SKMLWrapper._todense(self.model_.predict_proba(SKMLWrapper._todense(X), *args, **kwargs))
        
        n_instances, n_classes = probas.shape
        shaped_probas = [
            np.array([
                1-probas[:, i], probas[:, i]
            ]).reshape(2, n_instances).T for i in range(n_classes)
        ]

        return shaped_probas
    
    def set_params(self, **parameters):
        self.model_.set_params(**parameters)
    
    def get_params(self):
        return self.model_.get_params()


class MLEmbedding:
    def __init__(self, embedder=CLEMS(qp.error.ae, False, params=dict(n_jobs=6)), regressor=RandomForestRegressor(n_estimators=10, n_jobs=6), classifier=MLkNN(k=5)):
        self.embedder = embedder
        self.regressor = regressor
        self.classifier = classifier
        self.learner = EmbeddingClassifier(self.embedder, self.regressor, self.classifier)

    @staticmethod
    def _todense(A):
        aux = A
        if issparse(aux):
            aux = aux.todense()
        return np.array(aux)
    
    def fit(self, X, y):
        Xtr, ytr = MLEmbedding._todense(X), MLEmbedding._todense(y)
        return self.learner.fit(Xtr, ytr)
    
    def predict(self, instances):
        results = self.learner.predict(MLEmbedding._todense(instances))
        return MLEmbedding._todense(results)
    
    def predict_proba(self, instances):
        return MLEmbedding._todense(self.learner.predict_proba(MLEmbedding._todense(instances)))

    def get_params(self):
        params = {}

        for k, v in self.embedder.get_params().items():
            params[f"embedder__{k}"] = v
        
        for k, v in self.regressor.get_params().items():
            params[f"regressor__{k}"] = v
        
        for k, v in self.classifier.get_params().items():
            params[f"classifier__{k}"] = v
        
        return params
    
    def set_params(self, **params):
        self.embedder.set_params(**{k.removeprefix("embedder__"):v for k, v in params.items() if k.startswith("embedder__")})
        self.regressor.set_params(**{k.removeprefix("regressor__"):v for k, v in params.items() if k.startswith("regressor__")})
        self.classifier.set_params(**{k.removeprefix("classifier__"):v for k, v in params.items() if k.startswith("classifier__")})


class MLLabelClusterer:
    def __init__(self, classifier=MLkNN(k=5), clusterer=MatrixLabelSpaceClusterer(clusterer=KMeans(n_clusters=3))):
        self.classifier = classifier
        self.clusterer = clusterer
        self.learner = LabelSpacePartitioningClassifier(classifier=self.classifier, clusterer=self.clusterer, require_dense=[True, True])
    
    @staticmethod
    def _todense(A):
        aux = A
        if issparse(aux):
            aux = aux.todense()
        return np.array(aux)
    
    def fit(self, X, y):
        Xtr, ytr = MLLabelClusterer._todense(X), MLLabelClusterer._todense(y)
        return self.learner.fit(Xtr, ytr)
    
    def predict(self, instances):
        results = self.learner.predict(MLLabelClusterer._todense(instances))
        return MLLabelClusterer._todense(results)
    
    def predict_proba(self, instances):
        return MLLabelClusterer._todense(self.learner.predict_proba(MLLabelClusterer._todense(instances)))

    def get_params(self):
        params = {}

        for k, v in self.classifier.get_params().items():
            params[f"classifier__{k}"] = v
        
        for k, v in self.clusterer.get_params().items():
            params[f"clusterer__{k}"] = v
        
        return params
    
    def set_params(self, **params):
        self.clusterer.set_params(**{k:v for k, v in params.items() if k.startswith("clusterer__")})
        self.classifier.set_params(**{k.removeprefix("classifier__"):v for k, v in params.items() if k.startswith("classifier__")})


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
    
    def set_params(self, **parameters):
        params = {f"estimator__{k}":v for k, v in parameters.items()}
        # self.base.set_params(**params)
        self.meta.set_params(**params)

    def get_params(self, deep=True):
        return self.meta.get_params()


class MLStackedRegressor:
    def __init__(self, base_regressor=Ridge(normalize=True)):
        self.base = deepcopy(MultiOutputRegressor(base_regressor))
        self.meta = deepcopy(MultiOutputRegressor(base_regressor))

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
    
    def set_params(self, **parameters):
        # params = parameters
        # params = {f"estimator__{k}":v for k, v in parameters.items()}
        #self.base.set_params(**params)
        self.meta.set_params(**{f'estimator__{k.removeprefix("reg__")}':v for k,v, in parameters.items()})

    def get_params(self, deep=True):
        return {f'reg__{k.removeprefix("estimator__")}':v for k,v in self.meta.get_params().items()}


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

