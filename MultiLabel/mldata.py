from typing import List, Union

import numpy as np
from sklearn.model_selection import train_test_split

from quapy.data import LabelledCollection
from quapy.functional import artificial_prevalence_sampling

from skmultilearn.model_selection import iterative_train_test_split

from scipy.sparse import issparse, vstack

class MultilabelledCollection:
    def __init__(self, instances, labels):
        assert labels.ndim==2, f'data does not seem to be multilabel {labels}'
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
    def n_features(self):
        return self.instances.shape[1]

    @property
    def binary(self):
        return False

    def __gen_index(self):
        return np.arange(len(self))

    def sampling_multi_index(self, size, cat, prev=None):
        if prev is None:  # no prevalence was indicated; returns an index for uniform sampling
            return np.random.choice(len(self), size, replace=size > len(self))
        aux = LabelledCollection(self.__gen_index(), self.labels[:, cat], classes_=[0,1])
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

    def train_test_split(self, train_prop=0.6, random_state=None, iterative=True):
        if iterative:
            tr_docs, tr_labels, te_docs, te_labels = \
                iterative_train_test_split(self.instances, self.labels, test_size=1-train_prop)
        else:
            tr_docs, te_docs, tr_labels, te_labels = \
                train_test_split(self.instances, self.labels, train_size=train_prop, random_state=random_state)
        return MultilabelledCollection(tr_docs, tr_labels), MultilabelledCollection(te_docs, te_labels)

    def artificial_sampling_generator(self, sample_size, category, n_prevalences=101, repeats=1, min_df=-1, allow_replacement=False):
        # see artificial_sampling_index_generator
        for index in self.artificial_sampling_index_generator(sample_size, category, n_prevalences, repeats, min_df, allow_replacement):
            yield self.sampling_from_index(index)

    def artificial_sampling_index_generator(self, sample_size, category, n_prevalences=101, repeats=1, min_df=-1, allow_replacement=False):
        # only categories with a number of positive instances >= min_df will be submitted to the APP exploration
        # in case min_df == -1 (default), this check is not performed
        # the exploration of prevalence values will not surpass the current prevalence of the class (i.e., sampling
        # is only performed if it can be done without replacement) if allow_replacement=False (default)
        dimensions = 2
        cat_size = self.counts()[category]
        max_reachable_prevalence = cat_size / sample_size
        if min_df == -1 or cat_size >= min_df:
            for request_prev in artificial_prevalence_sampling(dimensions, n_prevalences, repeats).flatten():
                doable_with_replacement = request_prev == 0 or cat_size > 0
                if (allow_replacement and doable_with_replacement) or request_prev <= max_reachable_prevalence:
                    yield self.sampling_multi_index(sample_size, category, request_prev)

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

    # @property
    # def label_cardinality(self):
    #     return self.labels.sum()/len(self)

    @property
    def Xy(self):
        return self.instances, self.labels
    
    def __add__(self, other):
        if other is None:
            return self
        elif issparse(self.instances) and issparse(other.instances):
            join_instances = vstack([self.instances, other.instances])
        elif isinstance(self.instances, list) and isinstance(other.instances, list):
            join_instances = self.instances + other.instances
        elif isinstance(self.instances, np.ndarray) and isinstance(other.instances, np.ndarray):
            join_instances = np.concatenate([self.instances, other.instances])
        else:
            raise NotImplementedError('unsupported operation for collection types')
        labels = np.concatenate([self.labels, other.labels])
        return MultilabelledCollection(join_instances, labels)


class MultilingualLabelledCollection:
    def __init__(self, langs:List[str], labelledCollections:List[Union[LabelledCollection, MultilabelledCollection]]):
        assert len(langs) == len(labelledCollections), 'length mismatch for langs and labelledCollection lists'
        assert all(isinstance(lc, LabelledCollection) or all(isinstance(lc, MultilabelledCollection)) for lc in labelledCollections), \
            'unexpected type for labelledCollections'
        assert all(labelledCollections[0].classes_ == lc_i.classes_ for lc_i in labelledCollections[1:]), \
            'inconsistent classes found for some labelled collections'
        self.llc = {l: lc for l, lc in zip(langs, labelledCollections)}
        self.classes_=labelledCollections[0].classes_

    @classmethod
    def fromLangDict(cls, lang_labelledCollection:dict):
        return MultilingualLabelledCollection(*list(zip(*list(lang_labelledCollection.items()))))

    def langs(self):
        return list(sorted(self.llc.keys()))

    def __getitem__(self, lang)->LabelledCollection:
        return self.llc[lang]

    @classmethod
    def load(cls, path: str, loader_func: callable):
        return MultilingualLabelledCollection(*loader_func(path))

    def __len__(self):
        return sum(map(len, self.llc.values()))

    def prevalence(self):
        prev = np.asarray([lc.prevalence() * len(lc) for lc in self.llc.values()]).sum(axis=0)
        return prev / prev.sum()

    def language_prevalence(self):
        lang_count = np.asarray([len(self.llc[l]) for l in self.langs()])
        return lang_count / lang_count.sum()

    def counts(self):
        return np.asarray([lc.counts() for lc in self.llc.values()]).sum(axis=0)

    @property
    def n_classes(self):
        return len(self.classes_)

    @property
    def binary(self):
        return self.n_classes == 2

    def __check_langs(self, l_dict:dict):
        assert len(l_dict)==len(self.langs()), 'wrong number of languages'
        assert all(l in l_dict for l in self.langs()), 'missing languages in l_sizes'

    def __check_sizes(self, l_sizes: Union[int,dict]):
        assert isinstance(l_sizes, int) or isinstance(l_sizes, dict), 'unexpected type for l_sizes'
        if isinstance(l_sizes, int):
            return {l:l_sizes for l in self.langs()}
        self.__check_langs(l_sizes)
        return l_sizes

    def sampling_index(self, l_sizes: Union[int,dict], *prevs, shuffle=True):
        l_sizes = self.__check_sizes(l_sizes)
        return {l:lc.sampling_index(l_sizes[l], *prevs, shuffle=shuffle) for l,lc in self.llc.items()}

    def uniform_sampling_index(self, l_sizes: Union[int, dict]):
        l_sizes = self.__check_sizes(l_sizes)
        return {l: lc.uniform_sampling_index(l_sizes[l]) for l,lc in self.llc.items()}

    def uniform_sampling(self, l_sizes: Union[int, dict]):
        l_sizes = self.__check_sizes(l_sizes)
        return MultilingualLabelledCollection.fromLangDict(
            {l: lc.uniform_sampling(l_sizes[l]) for l,lc in self.llc.items()}
        )

    def sampling(self, l_sizes: Union[int, dict], *prevs, shuffle=True):
        l_sizes = self.__check_sizes(l_sizes)
        return MultilingualLabelledCollection.fromLangDict(
            {l: lc.sampling(l_sizes[l], *prevs, shuffle=shuffle) for l,lc in self.llc.items()}
        )

    def sampling_from_index(self, l_index:dict):
        self.__check_langs(l_index)
        return MultilingualLabelledCollection.fromLangDict(
            {l: lc.sampling_from_index(l_index[l]) for l,lc in self.llc.items()}
        )

    def split_stratified(self, train_prop=0.6, random_state=None):
        train, test = list(zip(*[self[l].split_stratified(train_prop, random_state) for l in self.langs()]))
        return MultilingualLabelledCollection(self.langs(), train), MultilingualLabelledCollection(self.langs(), test)

    def asLabelledCollection(self, return_langs=False):
        lXy_list = [([l]*len(lc),*lc.Xy) for l, lc in self.llc.items()]  # a list with (lang_i, Xi, yi)
        ls,Xs,ys = list(zip(*lXy_list))
        ls = np.concatenate(ls)
        vertstack = vstack if issparse(Xs[0]) else np.vstack
        Xs = vertstack(Xs)
        ys = np.concatenate(ys)
        lc = LabelledCollection(Xs, ys, classes_=self.classes_)
        # return lc, ls if return_langs else lc
