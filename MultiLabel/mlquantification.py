import math
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans

import sklearn.preprocessing
from scipy.sparse import issparse
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LassoCV, MultiTaskLassoCV, LassoLars, LassoLarsCV, \
    ElasticNet, MultiTaskElasticNetCV, MultiTaskElasticNet, LinearRegression, ARDRegression, BayesianRidge, SGDRegressor

from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.cluster import RandomLabelSpaceClusterer, MatrixLabelSpaceClusterer
from skmultilearn.problem_transform import LabelPowerset

from sklearn.feature_selection import chi2, SelectKBest
from sklearn.tree import DecisionTreeRegressor

import quapy as qp
from MultiLabel.mlclassification import MLStackedClassifier, MLStackedRegressor
from MultiLabel.mldata import MultilabelledCollection
from quapy.method.aggregative import CC, ACC, PACC, AggregativeQuantifier
from quapy.method.base import BaseQuantifier
from quapy.data import LabelledCollection

from abc import abstractmethod


class MLQuantifier:
    @abstractmethod
    def fit(self, data: MultilabelledCollection): ...

    @abstractmethod
    def quantify(self, instances): ...

    @abstractmethod
    def set_params(self, **parameters): ...

    @abstractmethod
    def get_params(self, deep=True): ...


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
        return self.learner.fit(*data.Xy)

    @abstractmethod
    def preclassify(self, instances): ...

    @abstractmethod
    def aggregate(self, predictions): ...

    def quantify(self, instances):
        predictions = self.preclassify(instances)
        return self.aggregate(predictions)
    
    def set_params(self, **parameters):
        self.learner.set_params(**parameters)

    def get_params(self, deep=True):
        return self.learner.get_params()


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


class DTWrapper(DecisionTreeRegressor):
    def predict_proba(self, X):
        return self.predict_proba(X)[:, :, 1]

class MLPCC(MLCC):
    def preclassify(self, instances):
        predictions = self.learner.predict_proba(instances)
        if issparse(predictions):
            predictions = predictions.toarray()
        if isinstance(predictions, list):
            predictions = np.asarray(predictions)[:, :, 1].T
        return predictions


class MLACC(MLCC):

    def fit(self, data:MultilabelledCollection, train_prop=0.6):
        self.classes_ = data.classes_
        train, val = data.train_test_split(train_prop=train_prop)
        self.learner.fit(*train.Xy)
        val_predictions = self.preclassify(val.instances)
        self.Pte_cond_estim_ = []
        for c in data.classes_:
            # pos_c = val.labels[:,c].sum()
            # neg_c = len(val) - pos_c
            Pmatrix = ACC.getPteCondEstim([0,1], val.labels[:,c], val_predictions[:,c])
            self.Pte_cond_estim_.append(Pmatrix)
        return self

    def preclassify(self, instances):
        predictions = self.learner.predict(instances)
        if issparse(predictions):
            predictions = predictions.toarray()
        return predictions

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
            Pmatrix = PACC.getPteCondEstim([0, 1], val.labels[:, c], c_posteriors)
            self.Pte_cond_estim_.append(Pmatrix)
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
        
        # def cat_job_with_mselection(args):
        #     trsp, valsp = args
        #     return deepcopy(self.q).fit(trsp, valsp)

        # if hasattr(self.q, "best_model"):
        #     trsp, valsp = data.train_test_split(train_prop=.6)
        #     self.estimators = qp.util.parallel(cat_job_with_mselection, zip(trsp.genLabelledCollections(), valsp.genLabelledCollections()), n_jobs=self.n_jobs)
        # else:
        self.estimators = qp.util.parallel(cat_job, data.genLabelledCollections(), n_jobs=self.n_jobs)

        # DEBUG: sequential run
        # self.estimators = []
        # for i, lc in enumerate(data.genLabelledCollections()):
        #     print(f"cat {i}")
        #     self.estimators.append(deepcopy(self.q).fit(lc))

        return self

    def quantify(self, instances):
        pos_prevs = np.zeros(len(self.classes_), dtype=float)
        for c in self.classes_:
            pos_prevs[c] = self.estimators[c].quantify(instances)[1]
        neg_prevs = 1-pos_prevs
        return np.asarray([neg_prevs, pos_prevs]).T
    
    def set_params(self, **parameters):
        self.q.set_params(**parameters)

    def get_params(self, deep=True):
        if self.estimators: # not nice, but required to save params in __main__
            return {f'estimator{i}':q.get_params() for i, q in enumerate(self.estimators)}
        return self.q.get_params()


class MLNaiveAggregativeQuantifier(MLNaiveQuantifier, MLAggregativeQuantifier):
    def __init__(self, q:AggregativeQuantifier, n_jobs=-1):
        # FIXME: assert below removed to nest MLGridSearchQ within the pipeline and avoid circular imports
        # assert isinstance(q, AggregativeQuantifier), 'the quantifier is not of type aggregative!'
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
                 estimator=MLNaiveQuantifier(CC(LinearSVC())),
                 regression='ridge',
                 protocol='npp',
                 n_samples=500,
                 sample_size=500,
                 norm=True,
                 means=True,
                 stds=True):

        assert protocol in ['npp', 'app'], 'unknown protocol'
        self.estimator = estimator
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

        self.trained_ = False
        self.reg_params_changed_ = False
        self.estimator_params_changed_ = False

    def set_params(self, **parameters):
        #FIXME: self.reg modifica los nombres de los parameters, comprobar
        paramse = {}
        paramsr = {}
        for k, v in parameters.items():
            if k.startswith("estimator__"):
                paramse[k.removeprefix("estimator__")] = v
            elif k.startswith("regressor__"):
                paramsr[k.removeprefix("regressor__")] = v

        if paramse:
            self.estimator_params_changed_ = True
            self.estimator.set_params(**paramse)
        if paramsr:
            self.reg_params_changed_ = True
            self.reg.set_params(**paramsr)
    
    def get_params(self):
        paramse = {f"estimator__{k}":v for k, v in self.estimator.get_params().items()}
        paramsr = {f"regressor__{k}":v for k, v in self.reg.get_params().items()}
        return paramse | paramsr

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
        prev = self.estimator.quantify(sample.instances)[:, 1]
        Xs.append(prev)
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
        tr, val = data.train_test_split(force_min_pos=2)
        assert all(tr.counts()>0), f'this is not gonna work, {tr.counts()}'

        if not self.trained_ or self.estimator_params_changed_:
            self.estimator.fit(tr)
        
        if not self.trained_ or self.estimator_params_changed_ or self.reg_params_changed_:
            if self.protocol == 'npp':
                Xs, ys = self.generate_samples_npp(val)
            elif self.protocol == 'app':
                Xs, ys = self.generate_samples_app(val)
            # Xs = self.norm.fit_transform(Xs)

            self.reg.fit(Xs, ys)

        self.trained_ = True
        self.reg_params_changed_ = False
        self.estimator_params_changed_ = False
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



class kMLQ(MLRegressionQuantification):
    class LiteMLPACC(MLPACC):
        def fit(self, train:MultilabelledCollection, val:MultilabelledCollection):
            self.classes_ = train.classes_
            self.learner.fit(*train.Xy)
            val_posteriors = self.preclassify(val.instances)
            self.Pte_cond_estim_ = []
            for c in train.classes_:
                pos_posteriors = val_posteriors[:, c]
                c_posteriors = np.asarray([1-pos_posteriors, pos_posteriors]).T
                Pmatrix = PACC.getPteCondEstim([0, 1], val.labels[:, c], c_posteriors)
                self.Pte_cond_estim.append(Pmatrix)
            return self

    def __init__(self, base, reg="ridge", k=3, protocol='app', error=qp.error.ae, n_samples=500, sample_size=500, norm=True, means=True, stds=True):
        assert protocol == "app", "npp not implemented"
        self.estimators_ = [deepcopy(kMLQ.LiteMLPACC(base)) for _ in range(k)]
        if reg == "ridge":
            self.reg = Ridge(normalize=norm)
        else:
            self.reg = reg
        self.k = k
        self.protocol = protocol
        self.error = error
        self.sample_size = sample_size
        self.n_samples = n_samples
        self.means = means
        self.stds = stds
    
    def _extract_features(self, sample, Xs, ys, samples_mean, samples_std):
        ys.append(sample.prevalence()[:, 1])
        prevs = np.hstack([estimator.quantify(sample.instances)[:, 1] for estimator in self.estimators_])
        Xs.append(prevs)

        if self.means:
            samples_mean.append(sample.instances.mean(axis=0).getA().flatten())
        if self.stds:
            samples_std.append(sample.instances.todense().std(axis=0).getA().flatten())
    
    def fit(self, data:MultilabelledCollection):
        self.classes_ = data.classes_
        tr, val = data.train_test_split()
        assert all(tr.counts() > 0), f'this is not gonna work, {tr.counts()}'
        self.max_drift_ = np.max(tr.counts())
        self.delta_ = self.max_drift_ / self.k

        ncats = len(self.classes_)
        nprevs = 21
        repeats = max(self.n_samples // (ncats * nprevs), 1)
        samples_idx_per_estimator = [list() for _ in range(self.k)]
        for cat in self.classes_:
            for sample_idx in tr.artificial_sampling_index_generator(
                sample_size=self.sample_size,
                category=cat,
                n_prevalences=nprevs,
                repeats=repeats,
                min_df=5
            ):
                sample = tr.sampling_from_index(sample_idx)
                current_drift = self.error(tr.prevalence()[:, 1], sample.prevalence()[:, 1])
                i = int(min(current_drift // self.delta_, len(self.estimators_)))
                samples_idx_per_estimator[i].append(sample_idx)
        
        for i in range(self.k):
            #Xs, ys = [], []
            samples = [tr.sampling_from_index(sample_idx) for sample_idx in samples_idx_per_estimator[i]]
            self.estimators_[i].fit(samples, val)
        
        Xs, ys = self.generate_samples_app(val)
        self.reg.fit(Xs, ys)
        return self
    
    def quantify(self, instances):
        Xs = np.hstack([estimator.quantify(instances)[:, 1] for estimator in self.estimators_])
        if self.means:
            sample_mean = instances.mean(axis=0).getA()
            Xs = np.hstack([Xs, sample_mean])
        if self.stds:
            sample_std = instances.todense().std(axis=0).getA()
            Xs = np.hstack([Xs, sample_std])
        Xs = self.reg.predict(Xs)
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
                 stds=True,
                 k=10):

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
        self.k = k
    
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

        while len(selected) < self.k and len(selected) < self.no_labels:
            corrs_sum = corrs[selected, :].sum(axis=0)
            new_sel = np.argmax(corrs_sum)
            if new_sel in selected:
                assert (corrs_sum <= 0).all(), "label already selected"
                break # in the event that there are no more labels with positive corr

            for s in selected:
                corrs[s, new_sel] = 0
                corrs[new_sel, s] = 0
            
            selected.append(new_sel)
        
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


class ClusterLabelPowersetQuantifier:
    def __init__(self, base=CC(LogisticRegression()), clusterer=MatrixLabelSpaceClusterer(clusterer=KMeans(n_clusters=5)), n_jobs=1):
        self.base = base
        self.clusterer = clusterer
        self.n_jobs = n_jobs
    
    def fit(self, data, **params):
        X, y = data.Xy
        self.n_cat = y.shape[1]

        # self.rc = self.clusterer.fit_predict(y, y)
        self.partitions = self.clusterer.fit_predict(X, y)

        def cat_job(p):
            lp = LabelPowerset()
            yt = lp.transform(y[:, p])
            q = deepcopy(self.base).fit(LabelledCollection(X, yt))
            return lp, q
        
        results = qp.util.parallel(cat_job, self.partitions, n_jobs=self.n_jobs)
        self.lps, self.qs = zip(*results)
    
    def quantify(self, X):
        yhat = np.zeros(self.n_cat)

        def cat_job(args):
            i, p = args
            yphat_perclass = self.qs[i].quantify(X)
            combinations = self.lps[i].inverse_transform(np.asarray(list(self.lps[i].unique_combinations_.values()))).todense()
            yphat = np.sum(np.multiply(combinations, yphat_perclass[:, np.newaxis]), axis=0)
            return p, yphat
        
        results = qp.util.parallel(cat_job, enumerate(self.partitions), n_jobs=self.n_jobs)
        for p, yphat in results:
            yhat[p] = yphat
        
        neg_prevs = 1 - yhat
        return np.asarray([neg_prevs, yhat]).T
    
    def get_params(self):
        paramsb = {f'base__{k}':v for k,v in self.base.get_params().items()}
        paramsc = {f'clusterer__{k}':v for k,v in self.clusterer.get_params().items()}
        return paramsb | paramsc
    
    def set_params(self, **params):
        paramsb = {k.removeprefix('base__'):v for k,v in params.items() if k.startswith('base__')}
        paramsc = {k.removeprefix('clusterer__'):v for k,v in params.items() if k.startswith('clusterer__')}
        self.base.set_params(**paramsb)
        self.clusterer.set_params(**paramsc)


# This class can be removed, ClusterLabelPowersetQuantifier does exactly the same
# if clusterer is set to skmultilearn.cluster.RandomLabelSpaceClusterer
class RakelDQuantifier(ClusterLabelPowersetQuantifier):
    def __init__(self, base=CC(LogisticRegression()), n_clusters=5, n_jobs=-1):
        self.base = base
        self.n_clusters = n_clusters
        self.n_jobs = n_jobs
        
    def fit(self, data, **params):
        X, y = data.Xy
        self.n_cat = y.shape[1]

        if self.n_cat < self.n_clusters:
            raise ValueError("the size of the cluster is not greater than 1")
        
        cluster_size = math.ceil(self.n_cat / self.n_clusters)
        self.clusterer = RandomLabelSpaceClusterer(cluster_size, self.n_clusters, allow_overlap=False)

        super().fit(data, **params)
        
        return self
    
    def get_params(self):
        params = self.base.get_params()
        params["n_clusters"] = self.n_clusters
        return params
    
    def set_params(self, **params):
        base_params = params
        if "n_clusters" in params.keys():
            base_params = {k:v for k, v in params.items() if k != "n_clusters"}
            self.n_clusters = params["n_clusters"]
        self.base.set_params(**base_params)