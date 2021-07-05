from copy import deepcopy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import quapy as qp
from functional import artificial_prevalence_sampling
from method.aggregative import PACC, CC, EMQ
from method.base import BaseQuantifier
from quapy.data import from_rcv2_lang_file, LabelledCollection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


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
        aux = LabelledCollection(self.__gen_index(), self.instances[:,cat])
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
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats):
            yield self.sampling(sample_size, category, prevs[1])

    def artificial_sampling_index_generator(self, sample_size, category, n_prevalences=101, repeats=1):
        dimensions = 2
        for prevs in artificial_prevalence_sampling(dimensions, n_prevalences, repeats):
            yield self.sampling_multi_index(sample_size, category, prevs[1])

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


class MultilabelQuantifier:
    def __init__(self, q:BaseQuantifier):
        self.q = q
        self.estimators = {}

    def fit(self, data:MultilabelledCollection):
        self.classes_ = data.classes_
        for cat, lc in enumerate(data.genLabelledCollections()):
            self.estimators[cat] = deepcopy(self.q).fit(lc)
        return self

    def quantify(self, instances):
        pos_prevs = np.zeros(len(self.classes_), dtype=float)
        for c in self.classes_:
            pos_prevs[c] = self.estimators[c].quantify(instances)[1]
        neg_prevs = 1-pos_prevs
        return np.asarray([neg_prevs, pos_prevs]).T


class MultilabelRegressionQuantification:
    def __init__(self, base_quantifier=CC(LinearSVC()), regression='ridge', n_samples=500, sample_size=500):
        self.estimator = MultilabelQuantifier(base_quantifier)
        self.regression = regression
        self.n_samples = n_samples
        self.sample_size = sample_size

    def fit(self, data:MultilabelledCollection):
        self.classes_ = data.classes_
        tr, te = data.train_test_split()
        self.estimator.fit(tr)
        Xs = []
        ys = []
        for sample in te.natural_sampling_generator(sample_size=self.sample_size, repeats=self.n_samples):
            ys.append(sample.prevalence()[:,1])
            Xs.append(self.estimator.quantify(sample.instances)[:,1])
        Xs = np.asarray(Xs)
        ys = np.asarray(ys)
        print(f'Xs in {Xs.shape}')
        print(f'ys in {ys.shape}')
        self.reg = Ridge().fit(Xs, ys) #normalize?
        return self

    def quantify(self, instances):
        Xs = self.estimator.quantify(instances)[:,1].reshape(1,-1)
        adjusted = self.reg.predict(Xs)
        adjusted = np.clip(adjusted, 0, 1)
        adjusted = adjusted.flatten()
        neg_prevs = 1-adjusted
        return np.asarray([neg_prevs, adjusted]).T



# read documents
path = f'./crosslingual_data/rcv12/en.small.txt'
docs, cats = from_rcv2_lang_file(path)

# split train-test
tr_docs, te_docs, tr_cats, te_cats = train_test_split(docs, cats, test_size=0.2, random_state=42)

# generate Y matrices
mlb = MultiLabelBinarizer()
ytr = mlb.fit_transform([cats.split(' ') for cats in tr_cats])
yte = mlb.transform([cats.split(' ') for cats in te_cats])
# retain 10 most populated categories
most_populated = np.argsort(ytr.sum(axis=0))[-10:]
ytr = ytr[:,most_populated]
yte = yte[:,most_populated]

tfidf = TfidfVectorizer(min_df=5)
Xtr = tfidf.fit_transform(tr_docs)
Xte = tfidf.transform(te_docs)

train = MultilabelledCollection(Xtr, ytr)
test = MultilabelledCollection(Xte, yte)

model = MultilabelQuantifier(PACC(LogisticRegression()))
model.fit(train)
estim_prevs = model.quantify(test.instances)
true_prevs = test.prevalence()
print('PACC:')
print(estim_prevs)
print(true_prevs)


model = MultilabelQuantifier(CC(LogisticRegression()))
model.fit(train)
estim_prevs = model.quantify(test.instances)
true_prevs = test.prevalence()
print('CC:')
print(estim_prevs)
print(true_prevs)


# model = MultilabelQuantifier(EMQ(LogisticRegression()))
# model.fit(train)
# estim_prevs = model.quantify(test.instances)
# true_prevs = test.prevalence()
# print('EMQ:')
# print(estim_prevs)
# print(true_prevs)

model = MultilabelRegressionQuantification(sample_size=200, n_samples=500)
model.fit(train)
estim_prevs = model.quantify(test.instances)
true_prevs = test.prevalence()
print('MRQ:')
print(estim_prevs)
print(true_prevs)

qp.environ['SAMPLE_SIZE']=100
mae = qp.error.mae(true_prevs, estim_prevs)
print(mae)



