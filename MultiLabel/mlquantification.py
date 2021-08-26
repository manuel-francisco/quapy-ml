import numpy as np
from copy import deepcopy

from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LassoCV, MultiTaskLassoCV, LassoLars, LassoLarsCV, \
    ElasticNet, MultiTaskElasticNetCV, MultiTaskElasticNet, LinearRegression, ARDRegression, BayesianRidge, SGDRegressor

import quapy as qp
from MultiLabel.mlclassification import MultilabelStackedClassifier
from MultiLabel.mldata import MultilabelledCollection
from method.aggregative import CC, ACC, PACC, AggregativeQuantifier
from method.base import BaseQuantifier

from abc import abstractmethod


class MLQuantifier:
    @abstractmethod
    def fit(self, data: MultilabelledCollection): ...

    @abstractmethod
    def quantify(self, instances): ...


class MLAggregativeQuantifier(MLQuantifier):
    def fit(self, data:MultilabelledCollection):
        self.learner.fit(*data.Xy)
        return self

    @abstractmethod
    def preclassify(self, instances): ...

    @abstractmethod
    def aggregate(self, predictions): ...

    def quantify(self, instances):
        predictions = self.preclassify(instances)
        return self.aggregate(predictions)


class MLCC(MLAggregativeQuantifier):
    def __init__(self, mlcls):
        self.learner = mlcls

    def preclassify(self, instances):
        return self.learner.predict(instances)

    def aggregate(self, predictions):
        pos_prev = predictions.mean(axis=0)
        neg_prev = 1 - pos_prev
        return np.asarray([neg_prev, pos_prev]).T


class MLPCC(MLCC):
    def __init__(self, mlcls):
        self.learner = mlcls

    def preclassify(self, instances):
        return self.learner.predict_proba(instances)


class MLACC(MLCC):
    def __init__(self, mlcls):
        self.learner = mlcls

    def fit(self, data:MultilabelledCollection, train_prop=0.6):
        self.classes_ = data.classes_
        train, val = data.train_test_split(train_prop=train_prop)
        self.learner.fit(*train.Xy)
        val_predictions = self.preclassify(val.instances)
        self.Pte_cond_estim_ = []
        for c in data.classes_:
            pos_c = val.labels[:,c].sum()
            neg_c = len(val) - pos_c
            self.Pte_cond_estim_.append(confusion_matrix(val.labels[:,c], val_predictions[:,c]).T / np.array([neg_c, pos_c]))
        return self

    def preclassify(self, instances):
        return self.learner.predict(instances)

    def aggregate(self, predictions):
        cc_prevs = super(MLACC, self).aggregate(predictions)
        acc_prevs = np.asarray([ACC.solve_adjustment(self.Pte_cond_estim_[c], cc_prevs[c]) for c in self.classes_])
        return acc_prevs


class MLPACC(MLPCC):
    def __init__(self, mlcls):
        self.learner = mlcls

    def fit(self, data:MultilabelledCollection, train_prop=0.6):
        self.classes_ = data.classes_
        train, val = data.train_test_split(train_prop=train_prop)
        self.learner.fit(*train.Xy)
        val_posteriors = self.preclassify(val.instances)
        self.Pte_cond_estim_ = []
        for c in data.classes_:
            pos_posteriors = val_posteriors[:,c]
            c_posteriors = np.asarray([1-pos_posteriors, pos_posteriors]).T
            self.Pte_cond_estim_.append(PACC.getPteCondEstim([0,1], val.labels[:,c], c_posteriors))
        return self

    def aggregate(self, posteriors):
        pcc_prevs = super(MLPACC, self).aggregate(posteriors)
        pacc_prevs = np.asarray([ACC.solve_adjustment(self.Pte_cond_estim_[c], pcc_prevs[c]) for c in self.classes_])
        return pacc_prevs


class MultilabelNaiveQuantifier(MLQuantifier):
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


class MultilabelNaiveAggregativeQuantifier(MultilabelNaiveQuantifier, MLAggregativeQuantifier):
    def __init__(self, q:AggregativeQuantifier, n_jobs=-1):
        assert isinstance(q, AggregativeQuantifier), 'the quantifier is not of type aggregative!'
        self.q = q
        self.estimators = None
        self.n_jobs = n_jobs

    def preclassify(self, instances):
        return np.asarray([q.preclassify(instances) for q in self.estimators]).swapaxes(0,1)

    def aggregate(self, predictions):
        pos_prevs = np.zeros(len(self.classes_), dtype=float)
        for c in self.classes_:
            pos_prevs[c] = self.estimators[c].aggregate(predictions[:,c])[1]
        neg_prevs = 1 - pos_prevs
        return np.asarray([neg_prevs, pos_prevs]).T

    def quantify(self, instances):
        predictions = self.preclassify(instances)
        return self.aggregate(predictions)


class MLRegressionQuantification:
    def __init__(self,
                 mlquantifier=MultilabelNaiveQuantifier(CC(LinearSVC())),
                 regression='ridge',
                 protocol='npp',
                 n_samples=500,
                 sample_size=500,
                 norm=True,
                 means=True,
                 stds=True):
        assert regression in ['ridge', 'svr'], 'unknown regression model'
        assert protocol in ['npp', 'app'], 'unknown protocol'
        self.estimator = mlquantifier
        if regression == 'ridge':
            self.reg = Ridge(normalize=norm)
        elif regression == 'svr':
            self.reg = MultiOutputRegressor(LinearSVR())
        self.protocol = protocol
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
        # self.covs = covs

    def _prepare_arrays(self, Xs, ys, samples_mean, samples_std):
        Xs = np.asarray(Xs)
        ys = np.asarray(ys)
        if self.means:
            samples_mean = np.asarray(samples_mean)
            Xs = np.hstack([Xs, samples_mean])
        if self.stds:
            samples_std = np.asarray(samples_std)
            Xs = np.hstack([Xs, samples_std])
        # if self.covs:

        return Xs, ys

    def generate_samples_npp(self, val):
        samples_mean = []
        samples_std = []
        Xs = []
        ys = []
        for sample in val.natural_sampling_generator(sample_size=self.sample_size, repeats=self.n_samples):
            ys.append(sample.prevalence()[:, 1])
            Xs.append(self.estimator.quantify(sample.instances)[:, 1])
            if self.means:
                samples_mean.append(sample.instances.mean(axis=0).getA().flatten())
            if self.stds:
                samples_std.append(sample.instances.todense().std(axis=0).getA().flatten())
        return self._prepare_arrays(Xs, ys, samples_mean, samples_std)

    def generate_samples_app(self, val):
        samples_mean = []
        samples_std = []
        Xs = []
        ys = []
        ncats = len(self.classes_)
        nprevs  = 21
        repeats = max(self.n_samples // (ncats * nprevs), 1)
        for cat in self.classes_:
            for sample in val.artificial_sampling_generator(sample_size=self.sample_size, category=cat, n_prevalences=nprevs, repeats=repeats):
                ys.append(sample.prevalence()[:, 1])
                Xs.append(self.estimator.quantify(sample.instances)[:, 1])
                if self.means:
                    samples_mean.append(sample.instances.mean(axis=0).getA().flatten())
                if self.stds:
                    samples_std.append(sample.instances.todense().std(axis=0).getA().flatten())
        return self._prepare_arrays(Xs, ys, samples_mean, samples_std)

    def fit(self, data:MultilabelledCollection):
        self.classes_ = data.classes_
        tr, val = data.train_test_split()
        self.estimator.fit(tr)
        if self.protocol == 'npp':
            Xs, ys = self.generate_samples_npp(val)
        elif self.protocol == 'app':
            Xs, ys = self.generate_samples_app(val)
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
        Xs = self.reg.predict(Xs)
        # Xs = self.norm.inverse_transform(Xs)
        adjusted = np.clip(Xs, 0, 1)
        adjusted = adjusted.flatten()
        neg_prevs = 1-adjusted
        return np.asarray([neg_prevs, adjusted]).T


# class