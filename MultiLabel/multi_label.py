from copy import deepcopy

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LassoCV, MultiTaskLassoCV, LassoLars, LassoLarsCV, \
    ElasticNet, MultiTaskElasticNetCV, MultiTaskElasticNet, LinearRegression, ARDRegression, BayesianRidge, SGDRegressor
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVC
from tqdm import tqdm

import quapy as qp
from functional import artificial_prevalence_sampling
from method.aggregative import PACC, CC, EMQ, PCC, ACC, HDy
from method.base import BaseQuantifier
from quapy.data import from_rcv2_lang_file, LabelledCollection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import numpy as np
from data.dataset  import Dataset




def cls():
    # return LinearSVC()
    return LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)


def calibratedCls():
    return CalibratedClassifierCV(cls())


class MultilabelledCollection:
    def __init__(self, instances, labels):
        assert labels.ndim==2, 'data does not seem to be multilabel'
        self.instances = instances
        self.labels = labels
        self.classes_ = np.arange(labels.shape[1])

    @classmethod
    def load(cls, path: str, loader_func: callable):
        return MultilabelledCollection(*loader_func(path))

    def __len__(self):
        return self.instances.shape[0]

    def prevalence(self):
        # return self.labels.mean(axis=0)
        pos = self.labels.mean(axis=0)
        neg = 1-pos
        return np.asarray([neg, pos]).T

    def counts(self):
        return self.labels.sum(axis=0)

    @property
    def n_classes(self):
        return len(self.classes_)

    @property
    def binary(self):
        return False

    def __gen_index(self):
        return np.arange(len(self))

    def sampling_multi_index(self, size, cat, prev=None):
        if prev is None:  # no prevalence was indicated; returns an index for uniform sampling
            return np.random.choice(len(self), size, replace=size>len(self))
        aux = LabelledCollection(self.__gen_index(), self.labels[:,cat])
        return aux.sampling_index(size, *[1-prev, prev])

    def uniform_sampling_multi_index(self, size):
        return np.random.choice(len(self), size, replace=size>len(self))

    def uniform_sampling(self, size):
        unif_index = self.uniform_sampling_multi_index(size)
        return self.sampling_from_index(unif_index)

    def sampling(self, size, category, prev=None):
        prev_index = self.sampling_multi_index(size, category, prev)
        return self.sampling_from_index(prev_index)

    def sampling_from_index(self, index):
        documents = self.instances[index]
        labels = self.labels[index, :]
        return MultilabelledCollection(documents, labels)

    def train_test_split(self, train_prop=0.6, random_state=None):
        tr_docs, te_docs, tr_labels, te_labels = \
            train_test_split(self.instances, self.labels, train_size=train_prop, random_state=random_state)
        return MultilabelledCollection(tr_docs, tr_labels), MultilabelledCollection(te_docs, te_labels)

    def artificial_sampling_generator(self, sample_size, category, n_prevalences=101, repeats=1):
        dimensions = 2
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats).flatten():
            yield self.sampling(sample_size, category, prevs)

    def artificial_sampling_index_generator(self, sample_size, category, n_prevalences=101, repeats=1):
        dimensions = 2
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats).flatten():
            yield self.sampling_multi_index(sample_size, category, prevs)

    def natural_sampling_generator(self, sample_size, repeats=100):
        for _ in range(repeats):
            yield self.uniform_sampling(sample_size)

    def natural_sampling_index_generator(self, sample_size, repeats=100):
        for _ in range(repeats):
            yield self.uniform_sampling_multi_index(sample_size)

    def asLabelledCollection(self, category):
        return LabelledCollection(self.instances, self.labels[:,category])

    def genLabelledCollections(self):
        for c in self.classes_:
            yield self.asLabelledCollection(c)

    @property
    def Xy(self):
        return self.instances, self.labels


class MultilabelClassifier:  # aka Funnelling Monolingual
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

class MLCC:
    def __init__(self, mlcls:MultilabelClassifier):
        self.mlcls = mlcls

    def fit(self, data:MultilabelledCollection):
        self.mlcls.fit(*data.Xy)

    def quantify(self, instances):
        pred = self.mlcls.predict(instances)
        pos_prev = pred.mean(axis=0)
        neg_prev = 1-pos_prev
        return np.asarray([neg_prev, pos_prev]).T


class MLPCC:
    def __init__(self, mlcls: MultilabelClassifier):
        self.mlcls = mlcls

    def fit(self, data: MultilabelledCollection):
        self.mlcls.fit(*data.Xy)

    def quantify(self, instances):
        pred = self.mlcls.predict_proba(instances)
        pos_prev = pred.mean(axis=0)
        neg_prev = 1 - pos_prev
        return np.asarray([neg_prev, pos_prev]).T


class MultilabelQuantifier:
    def __init__(self, q:BaseQuantifier, n_jobs=-1):
        self.q = q
        self.estimators = None
        self.n_jobs = n_jobs

    def fit(self, data:MultilabelledCollection):
        self.classes_ = data.classes_

        def cat_job(lc):
            return deepcopy(self.q).fit(lc)

        self.estimators = qp.util.parallel(cat_job, data.genLabelledCollections(), n_jobs=self.n_jobs)
        return self

    def quantify(self, instances):
        pos_prevs = np.zeros(len(self.classes_), dtype=float)
        for c in self.classes_:
            pos_prevs[c] = self.estimators[c].quantify(instances)[1]
        neg_prevs = 1-pos_prevs
        return np.asarray([neg_prevs, pos_prevs]).T


class MultilabelRegressionQuantification:
    def __init__(self, base_quantifier=CC(LinearSVC()), regression='ridge', n_samples=500, sample_size=500, norm=True,
                 means=True, stds=True):
        assert regression in ['ridge'], 'unknown regression model'
        self.estimator = MultilabelQuantifier(base_quantifier)
        if regression == 'ridge':
            self.reg = Ridge(normalize=norm)
        # self.reg = MultiTaskLassoCV(normalize=norm)
        # self.reg = KernelRidge(kernel='rbf')
        # self.reg = LassoLarsCV(normalize=norm)
        # self.reg = MultiTaskElasticNetCV(normalize=norm) <- bien
        #self.reg = LinearRegression(normalize=norm) # <- bien
        # self.reg = MultiOutputRegressor(ARDRegression(normalize=norm))  # <- bastante bien, incluso sin norm
        # self.reg = MultiOutputRegressor(BayesianRidge(normalize=False))  # <- bastante bien, incluso sin norm
        # self.reg = MultiOutputRegressor(SGDRegressor())  # lento, no va
        self.regression = regression
        self.n_samples = n_samples
        self.sample_size = sample_size
        # self.norm = StandardScaler()
        self.means = means
        self.stds = stds

    def fit(self, data:MultilabelledCollection):
        self.classes_ = data.classes_
        tr, te = data.train_test_split()
        self.estimator.fit(tr)
        samples_mean = []
        samples_std = []
        Xs = []
        ys = []
        for sample in te.natural_sampling_generator(sample_size=self.sample_size, repeats=self.n_samples):
            ys.append(sample.prevalence()[:,1])
            Xs.append(self.estimator.quantify(sample.instances)[:,1])
            if self.means:
                samples_mean.append(sample.instances.mean(axis=0).getA().flatten())
            if self.stds:
                samples_std.append(sample.instances.todense().std(axis=0).getA().flatten())
        Xs = np.asarray(Xs)
        ys = np.asarray(ys)
        if self.means:
            samples_mean = np.asarray(samples_mean)
            Xs = np.hstack([Xs, samples_mean])
        if self.stds:
            samples_std = np.asarray(samples_std)
            Xs = np.hstack([Xs, samples_std])
        # Xs = self.norm.fit_transform(Xs)
        self.reg.fit(Xs, ys)
        return self

    def quantify(self, instances):
        Xs = self.estimator.quantify(instances)[:,1].reshape(1,-1)
        if self.means:
            sample_mean = instances.mean(axis=0).getA()
            Xs = np.hstack([Xs, sample_mean])
        if self.stds:
            sample_std = instances.todense().std(axis=0).getA()
            Xs = np.hstack([Xs, sample_std])
        # Xs = self.norm.transform(Xs)
        adjusted = self.reg.predict(Xs)
        adjusted = np.clip(adjusted, 0, 1)
        adjusted = adjusted.flatten()
        neg_prevs = 1-adjusted
        return np.asarray([neg_prevs, adjusted]).T

sample_size = 250
n_samples = 1000

def models():
    yield 'CC', MultilabelQuantifier(CC(cls()))
    yield 'PCC', MultilabelQuantifier(PCC(cls()))
    yield 'MLCC', MLCC(MultilabelClassifier(cls()))
    yield 'MLPCC', MLPCC(MultilabelClassifier(cls()))
    # yield 'PACC', MultilabelQuantifier(PACC(cls()))
    # yield 'EMQ', MultilabelQuantifier(EMQ(calibratedCls()))
    common={'sample_size':sample_size, 'n_samples': n_samples, 'norm': True}
    # yield 'MRQ-CC', MultilabelRegressionQuantification(base_quantifier=CC(cls()), **common)
    yield 'MRQ-PCC', MultilabelRegressionQuantification(base_quantifier=PCC(cls()), **common)
    yield 'MRQ-PACC', MultilabelRegressionQuantification(base_quantifier=PACC(cls()), **common)


dataset = 'reuters21578'
data = Dataset.load(dataset, pickle_path=f'./pickles/{dataset}.pickle')

Xtr, Xte = data.vectorize()
ytr = data.devel_labelmatrix.todense().getA()
yte = data.test_labelmatrix.todense().getA()

most_populadted = np.argsort(ytr.sum(axis=0))[-25:]
ytr = ytr[:, most_populadted]
yte = yte[:, most_populadted]

train = MultilabelledCollection(Xtr, ytr)
test = MultilabelledCollection(Xte, yte)

print(f'Train-prev: {train.prevalence()[:,1]}')
print(f'Test-prev: {test.prevalence()[:,1]}')
print(f'MLPE: {qp.error.mae(train.prevalence(), test.prevalence()):.5f}')

# print('NPP:')
# test_indexes = list(test.natural_sampling_index_generator(sample_size=sample_size, repeats=100))
# for model_name, model in models():
#     model.fit(train)
#     errs = []
#     for index in test_indexes:
#         sample = test.sampling_from_index(index)
#         estim_prevs = model.quantify(sample.instances)
#         true_prevs = sample.prevalence()
#         errs.append(qp.error.mae(true_prevs, estim_prevs))
#     print(f'{model_name:10s}\tmae={np.mean(errs):.5f}')

print('APP:')
test_indexes = []
for cat in train.classes_:
    test_indexes.append(list(test.artificial_sampling_index_generator(sample_size=sample_size, category=cat, n_prevalences=21, repeats=10)))

for model_name, model in models():
    model.fit(train)
    macro_errs = []
    for cat_indexes in test_indexes:
        errs = []
        for index in cat_indexes:
            sample = test.sampling_from_index(index)
            estim_prevs = model.quantify(sample.instances)
            true_prevs = sample.prevalence()
            errs.append(qp.error.mae(true_prevs, estim_prevs))
        macro_errs.append(np.mean(errs))
    print(f'{model_name:10s}\tmae={np.mean(macro_errs):.5f}')



