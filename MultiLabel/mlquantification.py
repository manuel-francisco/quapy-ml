import numpy as np
from copy import deepcopy

import sklearn.preprocessing
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LassoCV, MultiTaskLassoCV, LassoLars, LassoLarsCV, \
    ElasticNet, MultiTaskElasticNetCV, MultiTaskElasticNet, LinearRegression, ARDRegression, BayesianRidge, SGDRegressor

from sklearn.feature_selection import chi2, SelectKBest

import quapy as qp
from MultiLabel.mlclassification import MLStackedClassifier, MLStackedRegressor
from MultiLabel.mldata import MultilabelledCollection
from quapy.method.aggregative import CC, ACC, PACC, AggregativeQuantifier
from quapy.method.base import BaseQuantifier

from abc import abstractmethod


class MLQuantifier:
    @abstractmethod
    def fit(self, data: MultilabelledCollection): ...

    @abstractmethod
    def quantify(self, instances): ...


class MLMLPE(MLQuantifier):
    def fit(self, data: MultilabelledCollection):
        self.tr_prev = data.prevalence()
        return self

    def quantify(self, instances):
        return self.tr_prev


class MLAggregativeQuantifier(MLQuantifier):
    def __init__(self, mlcls):
        self.learner = mlcls

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


def select_features(X, y):
    feature_scores = []
    for i in range(y.shape[1]):
        skb = SelectKBest(chi2, k="all")
        skb.fit(X, y[:, i])
        feature_scores.append(list(skb.scores_))
    
    return np.argsort(-np.mean(feature_scores, axis=0))





class MLCompositeAggregativeQuantifier(MLAggregativeQuantifier):
    def __init__(self, mlcls1, mlcls2, mlcls3):
        self.learner1 = mlcls1
        self.learner2 = mlcls2
        self.learner3 = mlcls3
        self.selected = None
        self.no_labels = 0
    
    def fit(self, data:MultilabelledCollection):
        self.no_labels = data.Xy[1].shape[1]
        corrs = np.corrcoef(data.Xy[1].T)
        np.fill_diagonal(corrs, 0)
        
        selected = []
        sel = np.argmax(corrs.sum(axis=0))
        sel_aux = np.argmax(corrs[sel, :])
        selected.append(sel)
        selected.append(sel_aux)

        while len(selected) < 10 and len(selected) < self.no_labels:
            selected.append(np.argmax(corrs[selected, :].sum(axis=0)))
        
        self.selected = sorted(selected)
        self.learner1.fit(data.Xy[0], data.Xy[1][:, self.selected])
        self.learner2.fit(*data.Xy)

        p1 = np.zeros((data.Xy[0].shape[0], self.no_labels))
        p1_aux = self.learner1.predict(data.Xy[0])
        p1[:, self.selected] = p1_aux
        p2 = self.learner2.predict(data.Xy[0])
        p = np.concatenate((p1, p2), axis=1)

        self.learner3.fit(p, data.Xy[1])

        return self


class MLCompositeCC(MLCompositeAggregativeQuantifier):
    def preclassify(self, instances):
        p1 = np.zeros((instances.shape[0], self.no_labels))
        p1_aux = self.learner1.predict(instances)
        p1[:, self.selected] = p1_aux
        p2 = self.learner2.predict(instances)

        p = np.concatenate((p1, p2), axis=1)
        return self.learner3.predict(p)
    
    def aggregate(self, predictions):
        pos_prev = predictions.mean(axis=0)
        neg_prev = 1 - pos_prev
        return np.asarray([neg_prev, pos_prev]).T


class MLSlicedAggregativeQuantifier(MLAggregativeQuantifier):
    def __init__(self, mlcls1, mlcls2):
        self.learner1 = mlcls1
        self.learner2 = mlcls2
        self.selected = None
        self.not_selected = None
        self.no_labels = 0
    
    def fit(self, data:MultilabelledCollection):
        self.no_labels = data.Xy[1].shape[1]
        corrs = np.corrcoef(data.Xy[1].T)
        np.fill_diagonal(corrs, 0)
        
        selected = []
        sel = np.argmax(corrs.sum(axis=0))
        sel_aux = np.argmax(corrs[sel, :])
        selected.append(sel)
        selected.append(sel_aux)

        while len(selected) < 10 and len(selected) < self.no_labels:
            selected.append(np.argmax(corrs[selected, :].sum(axis=0)))
        
        self.selected = sorted(selected)
        self.not_selected = [i for i in range(self.no_labels) if i not in self.selected]
        self.learner1.fit(data.Xy[0], data.Xy[1][:, self.selected])
        self.learner2.fit(data.Xy[0], data.Xy[1][:, self.not_selected])

        return self


class MLSlicedCC(MLSlicedAggregativeQuantifier):
    def preclassify(self, instances):
        p = np.zeros((instances.shape[0], self.no_labels))
        p1_aux = self.learner1.predict(instances)
        p2_aux = self.learner2.predict(instances)
        p[:, self.selected] = p1_aux
        p[:, self.not_selected] = p2_aux
        
        return p
    
    def aggregate(self, predictions):
        pos_prev = predictions.mean(axis=0)
        neg_prev = 1 -  pos_prev
        return np.asarray([neg_prev, pos_prev]).T



class MLCC(MLAggregativeQuantifier):
    def preclassify(self, instances):
        return self.learner.predict(instances)

    def aggregate(self, predictions):
        pos_prev = predictions.mean(axis=0)
        neg_prev = 1 - pos_prev
        return np.asarray([neg_prev, pos_prev]).T


class MLPCC(MLCC):
    def preclassify(self, instances):
        return self.learner.predict_proba(instances)


class MLACC(MLCC):

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


class MLNaiveQuantifier(MLQuantifier):
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


class MLNaiveAggregativeQuantifier(MLNaiveQuantifier, MLAggregativeQuantifier):
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
                 mlquantifier=MLNaiveQuantifier(CC(LinearSVC())),
                 regression='ridge',
                 protocol='npp',
                 n_samples=500,
                 sample_size=500,
                 norm=True,
                 means=True,
                 stds=True):

        assert protocol in ['npp', 'app'], 'unknown protocol'
        self.estimator = mlquantifier
        if isinstance(regression, str):
            assert regression in ['ridge', 'svr'], 'unknown regression model'
            if regression == 'ridge':
                self.reg = Ridge(normalize=norm)
            elif regression == 'svr':
                self.reg = MultiOutputRegressor(LinearSVR())
        else:
            self.reg = regression
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

    def _extract_features(self, sample, Xs, ys, samples_mean, samples_std):
        ys.append(sample.prevalence()[:, 1])
        Xs.append(self.estimator.quantify(sample.instances)[:, 1])
        if self.means:
            samples_mean.append(sample.instances.mean(axis=0).getA().flatten())
        if self.stds:
            samples_std.append(sample.instances.todense().std(axis=0).getA().flatten())

    def generate_samples_npp(self, val):
        Xs, ys = [], []
        samples_mean, samples_std = [], []
        for sample in val.natural_sampling_generator(sample_size=self.sample_size, repeats=self.n_samples):
            self._extract_features(sample, Xs, ys, samples_mean, samples_std)
        return self._prepare_arrays(Xs, ys, samples_mean, samples_std)


    def generate_samples_app(self, val):
        Xs, ys = [], []
        samples_mean, samples_std = [], []
        ncats = len(self.classes_)
        nprevs  = 21
        repeats = max(self.n_samples // (ncats * nprevs), 1)
        for cat in self.classes_:
            for sample in val.artificial_sampling_generator(
                    sample_size=self.sample_size, category=cat, n_prevalences=nprevs, repeats=repeats, min_df=5):
                self._extract_features(sample, Xs, ys, samples_mean, samples_std)
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


class CompositeMLRegressionQuantification(MLRegressionQuantification):
    def __init__(self,
                 mlquantifier=MLNaiveQuantifier(CC(LogisticRegression())),
                 bquantifier=MLNaiveAggregativeQuantifier(CC(LogisticRegression())),
                 regression='ridge',
                 protocol='npp',
                 n_samples=500,
                 sample_size=500,
                 norm=True,
                 means=True,
                 stds=True):

        assert protocol in ['npp', 'app'], 'unknown protocol'
        self.estimator = mlquantifier
        self.binary_estimator = bquantifier
        if isinstance(regression, str):
            assert regression in ['ridge', 'svr'], 'unknown regression model'
            if regression == 'ridge':
                self.reg = Ridge(normalize=norm)
            elif regression == 'svr':
                self.reg = MultiOutputRegressor(LinearSVR())
        else:
            self.reg = regression
        
        self.protocol = protocol
        self.regression = regression
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.means = means
        self.stds = stds

        self.selected = None
        self.not_selected = None
    
    def _extract_features(self, sample, Xs, ys, samples_mean, samples_std):
        ys.append(sample.prevalence()[:, 1])

        Xs1 = self.estimator.quantify(sample.instances)[:, 1]
        Xs2 = self.binary_estimator.quantify(sample.instances)[:, 1]
        Xsi = np.zeros((1, self.no_labels))
        Xsi[0, self.selected] = Xs1
        Xsi[0, self.not_selected] = Xs2

        adjusted = np.clip(Xsi, 0, 1)
        adjusted = adjusted.flatten()
        
        Xs.append(adjusted)
        if self.means:
            samples_mean.append(sample.instances.mean(axis=0).getA().flatten())
        if self.stds:
            samples_std.append(sample.instances.todense().std(axis=0).getA().flatten())
    
    def fit(self, data:MultilabelledCollection):
        self.classes_ = data.classes_


        self.no_labels = data.Xy[1].shape[1]
        corrs = np.corrcoef(data.Xy[1].T)
        np.fill_diagonal(corrs, 0)
        
        selected = []
        sel = np.argmax(corrs.sum(axis=0))
        sel_aux = np.argmax(corrs[sel, :])
        selected.append(sel)
        selected.append(sel_aux)
        corrs[sel, sel_aux] = 0
        corrs[sel_aux, sel] = 0

        while len(selected) < 10 and len(selected) < self.no_labels:
            new_sel = np.argmax(corrs[selected, :].sum(axis=0))
            assert not new_sel in selected, "already selected"
            selected.append(new_sel)

            for i in range(len(selected)):
                for j in range(i, len(selected)):
                    corrs[selected[i], selected[j]] = 0
                    corrs[selected[j], selected[i]] = 0
        
        self.selected = sorted(selected)
        self.not_selected = [i for i in range(self.no_labels) if i not in self.selected]

        # p1 = np.zeros((data.Xy[0].shape[0], self.no_labels))
        # p1_aux = self.learner1.predict(data.Xy[0])
        # p1[:, self.selected] = p1_aux
        # p2 = self.learner2.predict(data.Xy[0])
        # p = np.concatenate((p1, p2), axis=1)


        tr, val = data.train_test_split()
        trX, trY = tr.Xy[0], tr.Xy[1]
        trML = MultilabelledCollection(trX, trY[:, self.selected])
        trB = MultilabelledCollection(trX, trY[:, self.not_selected])

        self.estimator.fit(trML)
        self.binary_estimator.fit(trB)
        if self.protocol == 'npp':
            Xs, ys = self.generate_samples_npp(val)
        elif self.protocol == 'app':
            Xs, ys = self.generate_samples_app(val)
        # Xs = self.norm.fit_transform(Xs)
        self.reg.fit(Xs, ys)
        return self
    
    def quantify(self, instances):
        Xs1 = self.estimator.quantify(instances)[:,1].reshape(1,-1)
        Xs2 = self.binary_estimator.quantify(instances)[:,1].reshape(1,-1)
        Xs = np.zeros((1, self.no_labels))
        Xs[0, self.selected] = Xs1
        Xs[0, self.not_selected] = Xs2

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


class StackMLRQuantifier:
    def __init__(self,
                 mlquantifier=MLNaiveQuantifier(CC(LinearSVC())),
                 regression='ridge',
                 protocol='npp',
                 n_samples=500,
                 sample_size=500,
                 norm=True,
                 means=True,
                 stds=True):
        if regression == 'ridge':
            reg = MLStackedRegressor(Ridge(normalize=True))
        elif regression == 'svr':
            reg = MLStackedRegressor(MultiOutputRegressor(LinearSVR()))
        else:
            ValueError(f'unknown regressor {regression}')

        self.base = MLRegressionQuantification(
            mlquantifier=mlquantifier,
            regression=reg,
            protocol=protocol,
            n_samples=n_samples,
            sample_size=sample_size,
            norm=norm,
            means=means,
            stds=stds)

    def fit(self, data:MultilabelledCollection):
        self.classes_ = data.classes_
        self.base.fit(data)
        return self

    def quantify(self, instances):
        return self.base.quantify(instances)


class MLadjustedCount(MLAggregativeQuantifier):
    def __init__(self, learner):
        self.learner = learner

    def preclassify(self, instances):
        return self.learner.predict(instances)

    def fit(self, data: MultilabelledCollection, train_prop=0.6):
        self.classes_ = data.classes_
        train, val = data.train_test_split(train_prop=train_prop)
        self.learner.fit(*train.Xy)
        val_predictions = self.preclassify(val.instances)
        val_true = val.labels

        N = len(val)
        C = val_predictions.T.dot(val_true) / N  # join probabilities [[P(y1,\hat{y}1), P(y2,\hat{y}1)], ... ]
        priorP = val_predictions.mean(axis=0).reshape(-1,1)  # priors [P(hat{y}1), P(hat{y}2), ...]
        self.Pte_cond_estim_ = np.true_divide(C, priorP, where=priorP>0)  # cond probabilities [[P(y1|\hat{y}1), P(y2|\hat{y}1)], ... ]

        return self

    def aggregate(self, predictions):
        P = sklearn.preprocessing.normalize(predictions, norm='l1')
        correction = P.dot(self.Pte_cond_estim_)
        adjusted = correction.mean(axis=0)
        return np.asarray([1-adjusted, adjusted]).T


class MLprobAdjustedCount(MLAggregativeQuantifier):
    def __init__(self, learner):
        self.learner = learner

    def preclassify(self, instances):
        return self.learner.predict_proba(instances)

    def fit(self, data: MultilabelledCollection, train_prop=0.6):
        self.classes_ = data.classes_
        train, val = data.train_test_split(train_prop=train_prop)
        self.learner.fit(*train.Xy)
        val_predictions = self.preclassify(val.instances)
        val_true = val.labels

        N = len(val)

        C = (val_predictions>0.5).T.dot(val_true) / N  # join probabilities [[P(y1,\hat{y}1), P(y2,\hat{y}1)], ... ]
        # not sure...

        priorP = val_predictions.mean(axis=0).reshape(-1,1)  # priors [P(hat{y}1), P(hat{y}2), ...]
        self.Pte_cond_estim_ = np.true_divide(C, priorP, where=priorP>0)  # cond probabilities [[P(y1|\hat{y}1), P(y2|\hat{y}1)], ... ]

        return self

    def aggregate(self, predictions):
        P = sklearn.preprocessing.normalize(predictions, norm='l1')
        correction = P.dot(self.Pte_cond_estim_)
        adjusted = correction.mean(axis=0)
        return np.asarray([1-adjusted, adjusted]).T
