import numpy as np
from sklearn.model_selection import train_test_split

from quapy.data import LabelledCollection
from quapy.functional import artificial_prevalence_sampling


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
        labels = self.labels[index]
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