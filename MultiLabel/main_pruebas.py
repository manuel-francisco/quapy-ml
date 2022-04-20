import signal
import traceback

signal.signal(signal.SIGUSR1, lambda sig, stack: traceback.print_stack(stack))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



import argparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, MultiTaskLasso
import itertools
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier, RegressorChain
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from skmultilearn.dataset import load_dataset, available_data_sets
from skmultilearn.ensemble import RakelD
from scipy.sparse import csr_matrix
import quapy as qp
from MultiLabel.mlclassification import MLLabelClusterer, MLStackedClassifier, MLGeneralStackedClassifier, LabelSpacePartion, MLStackedRegressor, MLTwinSVM, MLknn, SKMLWrapper, MLEmbedding
from MultiLabel.mldata import MultilabelledCollection
from MultiLabel.mlquantification import MLCompositeCC, MLNaiveQuantifier, MLCC, MLPCC, MLRegressionQuantification, \
    MLACC, \
    MLPACC, MLNaiveAggregativeQuantifier, MLMLPE, MLSlicedCC, RakelDQuantifier, StackMLRQuantifier, MLadjustedCount, MLprobAdjustedCount, \
    CompositeMLRegressionQuantification, MLAggregativeQuantifier, MLQuantifier, ClusterLabelPowersetQuantifier
from MultiLabel.mlmodel_selection import MLGridSearchQ
from quapy.method.aggregative import PACC, CC, EMQ, PCC, ACC, HDy
import numpy as np
from data.dataset  import Dataset
from mlevaluation import ml_natural_prevalence_prediction, ml_artificial_prevalence_prediction
import sys
import os
import pickle

import random

from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN, MLARAM, MLTSVM
from quapy.model_selection import GridSearchQ

from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV, Lasso, LassoCV, MultiTaskLassoCV, LassoLars, LassoLarsCV, \
    ElasticNet, MultiTaskElasticNetCV, MultiTaskElasticNet, LinearRegression, ARDRegression, BayesianRidge, SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


seed = 42
random.seed(seed)
np.random.seed(seed)


def cls():
    return LogisticRegression(max_iter=2000, solver='lbfgs')


def calibratedCls():
    return CalibratedClassifierCV(cls())

sample_size = 100 #MIRAR run_experiment
n_samples = 5000 #MIRAR run_experiment

picklepath = 'pickles_manuel'

#SKMULTILEARN_ALL_DATASETS = sorted(set([x[0] for x in available_data_sets().keys()]))
SKMULTILEARN_ALL_DATASETS = ['Corel5k', 'bibtex', 'birds', 'delicious', 'emotions', 'enron', 'genbase', 'mediamill', 'medical', 'scene', 'tmc2007_500', 'yeast']
SKMULTILEARN_NOBIG_DATASETS = ['birds', 'emotions', 'enron', 'genbase', 'mediamill', 'medical', 'scene', 'tmc2007_500', 'yeast']
SKMULTILEARN_SMALL_DATASETS = ['birds', 'emotions', 'enron', 'genbase', 'medical', 'scene', 'tmc2007_500', 'yeast'] #offline
SKMULTILEARN_RED_DATASETS = [x+'-red' for x in SKMULTILEARN_ALL_DATASETS]
TC_DATASETS = ['reuters21578', 'jrcall', 'ohsumed', 'rcv1']
TC_DATASETS_REDUCED = ['rcv1', 'ohsumed']


SPLITS = [ # WARNING: sin jrcall ni rcv1
    ['mediamill', 'ohsumed', 'tmc2007_500', 'reuters21578'],
    ['Corel5k', 'yeast', 'enron', 'delicious', 'genbase'], 
    ['medical', 'scene', 'emotions', 'birds', 'bibtex'],
]


ALLTABLE = SKMULTILEARN_ALL_DATASETS + TC_DATASETS

#DATASETS = ['reuters21578']
COREL5K = ['Corel5k']



def models(subset, n_prevalences=101, repeats=25): # CAMBIAR EN __main__
    def select_best(model, param_grid=None, n_jobs=-1, single=False):
        if not param_grid:
            param_grid = dict(
                #C=np.array([1., 1.e-03, 1.e-02, 1.e-01, 1.e+01, 1.e+02, 1.e+03]), # np.logspace(-3, 3, 7) but with the 1. first
                C=np.array([1., 10, 100, 1000, .1]), # np.logspace(-3, 3, 7) but with the 1. first
                class_weight=["balanced", None],
            )
        
        if single:
            return GridSearchQ(
                model=model,
                param_grid=param_grid,
                sample_size=100,
                n_jobs=1,
                verbose=True,
                n_prevpoints=n_prevalences,
                protocol='app',
                n_repetitions=repeats,
            )
        else:
            return MLGridSearchQ(
                model=model,
                param_grid=param_grid,
                sample_size=100,
                n_jobs=n_jobs,
                verbose=True,
                n_prevalences=n_prevalences,
                repeats=repeats,
            )
    

    if subset == "general" or subset == "all":
        # common={'sample_size': sample_size, 'n_samples': n_samples, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}

        # naives (Binary Classification + Binary Quantification)
        yield 'NaiveCC', MLNaiveAggregativeQuantifier(select_best(CC(cls()), single=True))
        yield 'NaivePCC', MLNaiveAggregativeQuantifier(select_best(PCC(cls()), single=True))
        yield 'NaiveACC', MLNaiveAggregativeQuantifier(select_best(ACC(cls()), single=True))
        yield 'NaivePACC', MLNaiveAggregativeQuantifier(select_best(PACC(cls()), single=True))
        # yield 'NaiveHDy', MLNaiveAggregativeQuantifier(select_best(HDy(cls()), single=True))
        # yield 'NaiveSLDNoCalibrado', MLNaiveAggregativeQuantifier(select_best(EMQ(cls()), single=True))
        # yield 'NaiveSLD', MLNaiveAggregativeQuantifier(select_best(EMQ(calibratedCls()), param_grid=dict(
        #         base_estimator__C=np.array([1., 1.e-03, 1.e-02, 1.e-01, 1.e+01, 1.e+02, 1.e+03]),
        #         base_estimator__class_weight=["balanced", None],
        #     ), single=True))

        # yield 'StackCC', select_best(MLCC(MLStackedClassifier(cls())))
        # yield 'StackPCC', select_best(MLPCC(MLStackedClassifier(cls())))
        # yield 'StackACC', select_best(MLACC(MLStackedClassifier(cls())))
        # yield 'StackPACC', select_best(MLPACC(MLStackedClassifier(cls())))

        CVStack_grid = {
            'meta__estimator__C': np.array([1., 10, 100, 1000, .1]),
            'meta__estimator__class_weight': ["balanced", None],
            'norm': [True, False],
        }

        yield 'CVStackCC', select_best(MLCC(MLGeneralStackedClassifier(cls(), cv=5, norm=True, passthrough=True)), param_grid=CVStack_grid)
        yield 'CVStackPCC', select_best(MLPCC(MLGeneralStackedClassifier(cls(), cv=5, norm=True, passthrough=True)), param_grid=CVStack_grid)
        yield 'CVStackACC', select_best(MLACC(MLGeneralStackedClassifier(cls(), cv=5, norm=True, passthrough=True)), param_grid=CVStack_grid)
        yield 'CVStackPACC', select_best(MLPACC(MLGeneralStackedClassifier(cls(), cv=5, norm=True, passthrough=True)), param_grid=CVStack_grid)

        common={'protocol':'app', 'sample_size':100, 'n_samples': 5000, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}
        MRQ_grid = {
            'regressor__estimator__C': np.array([1., 10, 100, 1000, .1]),
        }

        yield 'MRQ-CC', select_best(MLRegressionQuantification(MLNaiveQuantifier(select_best(CC(cls()), single=True)), **common), param_grid=MRQ_grid)
        yield 'MRQ-PCC', select_best(MLRegressionQuantification(MLNaiveQuantifier(select_best(PCC(cls()), single=True)), **common), param_grid=MRQ_grid)
        yield 'MRQ-ACC', select_best(MLRegressionQuantification(MLNaiveQuantifier(select_best(ACC(cls()), single=True)), **common), param_grid=MRQ_grid)
        yield 'MRQ-PACC', select_best(MLRegressionQuantification(MLNaiveQuantifier(select_best(PACC(cls()), single=True)), **common), param_grid=MRQ_grid)
        # yield 'MRQ-StackCC', MLRegressionQuantification(select_best(MLCC(MLStackedClassifier(cls()))), **common)
        # yield 'MRQ-StackPCC', MLRegressionQuantification(select_best(MLPCC(MLStackedClassifier(cls()))), **common)
        # yield 'MRQ-StackACC', MLRegressionQuantification(select_best(MLACC(MLStackedClassifier(cls()))), **common)
        # yield 'MRQ-StackPACC', MLRegressionQuantification(select_best(MLPACC(MLStackedClassifier(cls()))), **common)

        MRQ_grid = {
            'estimator__norm': [True, False], #for MLGeneralStackedClassifier
            'estimator__meta__estimator__C': np.array([1., 10, 100, 1000, .1]),
            'estimator__meta__estimator__class_weight': ["balanced", None],
            'regressor__estimator__C': np.array([1., 10, 100, 1000, .1]),
        }

        yield 'MRQ-CVStackCC', select_best(MLRegressionQuantification(MLCC(MLGeneralStackedClassifier(cls(), cv=5, norm=True, passthrough=True)), **common), param_grid=MRQ_grid)
        yield 'MRQ-CVStackPCC', select_best(MLRegressionQuantification(MLPCC(MLGeneralStackedClassifier(cls(), cv=5, norm=True, passthrough=True)), **common), param_grid=MRQ_grid)
        yield 'MRQ-CVStackACC', select_best(MLRegressionQuantification(MLACC(MLGeneralStackedClassifier(cls(), cv=5, norm=True, passthrough=True)), **common), param_grid=MRQ_grid)
        yield 'MRQ-CVStackPACC', select_best(MLRegressionQuantification(MLPACC(MLGeneralStackedClassifier(cls(), cv=5, norm=True, passthrough=True)), **common), param_grid=MRQ_grid)
    
    if subset == "mlc" or subset == "all":
        #MLC experiments
        yield "MLkNN-MLPCC", select_best(MLPCC(SKMLWrapper(MLkNN())), param_grid={
            'k': range(1,10,2),
            's': [0.5, 0.7, 1.0]
        })
        yield 'ChainPCC', select_best(MLPCC(ClassifierChain(cls())), param_grid={
            'base_estimator__C': np.logspace(-3, 3, 7),
            'base_estimator__class_weight': [None, "balanced"],
        })
        yield 'CLEMS-PCC', select_best(MLPCC(MLEmbedding()), param_grid={
            'regressor__n_estimators': [10, 20, 50],
            'classifier__k': range(1, 10, 2),
            'classifier__s': [.5, .7, 1.],
        }, n_jobs=1)
        yield 'LClusterer-PCC', select_best(MLPCC(MLLabelClusterer()), param_grid={
            # 'classifier__k': range(1,10,2),
            # 'classifier__s': [0.5, 0.7, 1.0],
            'clusterer__n_clusters': [2, 3, 5, 10, 50],#, 100], #segfault for 100
        }, n_jobs=6)
        yield 'DT-PCC', select_best(MLPCC(DecisionTreeClassifier()), param_grid={
            'criterion': ["gini", "entropy"],
            #'classifier__class_weight': [None, "balanced"],
        })
        yield 'RF-PCC', select_best(MLPCC(RandomForestClassifier(n_jobs=6)), param_grid={
            'n_estimators': [10, 100, 200],
            #'classifier__criterion': ["gini", "entropy"],
        }, n_jobs=1)
    

    if subset == "mlq" or subset == "all":
        common={'protocol':'app', 'sample_size':100, 'n_samples': 5000, 'norm': True, 'means':False, 'stds':False}
        yield 'RakelD-PCC', select_best(RakelDQuantifier(base=PCC(cls())), param_grid={
            'n_clusters': [2, 5, 10, 50, 100],
            'C': np.array([1., 10, 100, 1000, .1]),
            'class_weight': ["balanced", None],
        })

        yield 'KMeansClustersPowerSet-PCC', select_best(ClusterLabelPowersetQuantifier(base=PCC(cls())), param_grid={
            'base__C': np.array([1., 10, 100, 1000, .1]),
            'base__class_weight': ["balanced", None],
            'clusterer__clusterer__n_clusters': [5, 15, 50, 100]
        })
        
        # Params here get a little weird because of MultiOutputRegressors and/or AdaBoost
        # yield 'MRQ-MultitaskLassoCV', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=MultiTaskLassoCV(normalize=True), **common), param_grid={
        #     # 'regressor__alpha': np.linspace(0.001, 20, .2),
        #     'estimator__C': np.array([1., 10, 100, 1000, .1]),
        #     'estimator__class_weight': ["balanced", None],
        # })
        # yield 'MRQ-RidgeCV', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=RidgeCV(normalize=True), **common), param_grid={
        #     # 'regressor__alpha': np.logspace(-6, 6, 13), # from https://scikit-learn.org/stable/modules/linear_model.html#ridge-complexity
        #     'estimator__C': np.array([1., 10, 100, 1000, .1]),
        #     'estimator__class_weight': ["balanced", None],
        # })
        # yield 'MRQ-LinearRegression', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=LinearRegression(n_jobs=-1), **common), param_grid={
        #     'estimator__C': np.array([1., 10, 100, 1000, .1]),
        #     'estimator__class_weight': ["balanced", None],
        # })
        yield 'MRQ-Ridge', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=Ridge(normalize=True), **common), param_grid={
            # 'regressor__alpha': np.logspace(-6, 6, 13), # from https://scikit-learn.org/stable/modules/linear_model.html#ridge-complexity
            'regressor__alpha': np.logspace(-3, 3, 7), # defaults https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html?highlight=ridge#sklearn.linear_model.RidgeCV
            'estimator__C': np.array([1., 10, 100, 1000, .1]),
            'estimator__class_weight': ["balanced", None],
        })
        yield 'MRQ-MultitaskLasso', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=MultiTaskLasso(normalize=True), **common), param_grid={
            'regressor__alpha': np.logspace(-3, 3, 7), #np.linspace(0.001, 20, .2),
            'estimator__C': np.array([1., 10, 100, 1000, .1]),
            'estimator__class_weight': ["balanced", None],
        })
        yield 'MRQ-RandomForest', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=RandomForestRegressor(n_jobs=6), **common), param_grid={
            "regressor__n_estimators": [10, 100, 200],
            'estimator__C': np.array([1., 10, 100, 1000, .1]),
            'estimator__class_weight': ["balanced", None],
        }, n_jobs=-1)
        # yield 'MRQ-AdaBoostChain', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=RegressorChain(AdaBoostRegressor()), **common), param_grid={
        #     'regressor__base_estimator__n_estimators': [10, 50, 100, 200],
        #     'estimator__C': np.array([1., 10, 100, 1000, .1]),
        #     'estimator__class_weight': ["balanced", None],
        # })
        # yield 'MRQ-AdaBoostStack', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=MLStackedRegressor(AdaBoostRegressor()), **common), param_grid={
        #     'regressor__estimator__n_estimators': [10, 50, 100, 200],
        #     'estimator__C': np.array([1., 10, 100, 1000, .1]),
        #     'estimator__class_weight': ["balanced", None],
        # })
        
        yield 'MRQ-LinearSVR', select_best(MLRegressionQuantification(select_best(MLPCC(MLStackedClassifier(cls()))), regression=MultiOutputRegressor(LinearSVR())), param_grid={
            "regressor__estimator__C": np.array([1., 10, 100, 1000, .1]),
        })
        yield 'MRQ-StackedLinearSVR', select_best(MLRegressionQuantification(MLNaiveQuantifier(select_best(PCC(cls()))), regression=MLStackedRegressor(LinearSVR())), param_grid={
            "regressor__C": np.array([1., 10, 100, 1000, .1]),
        })
        yield 'MRQ-ChainedLinearSVR', select_best(MLRegressionQuantification(MLNaiveQuantifier(select_best(PCC(cls()))), regression=RegressorChain(LinearSVR())), param_grid={
            "regressor__base_estimator__C": np.array([1., 10, 100, 1000, .1]),
        })

    return None




def get_dataset(dataset_name, dopickle=True):
    datadir = f'{qp.util.get_quapy_home()}/pickles'
    datapath = f'{datadir}/{dataset_name}.pkl'
    if dopickle:
        if os.path.exists(datapath):
            print(f'returning pickled object in {datapath}')
            return pickle.load(open(datapath, 'rb'))

    if dataset_name in SKMULTILEARN_ALL_DATASETS + SKMULTILEARN_RED_DATASETS:
        clean_name = dataset_name.replace('-red','')
        Xtr, ytr, feature_names, label_names = load_dataset(clean_name, 'train')
        Xte, yte, _, _ = load_dataset(clean_name, 'test')
        print(f'n-labels = {len(label_names)}')

        Xtr = csr_matrix(Xtr)
        Xte = csr_matrix(Xte)

        ytr = ytr.todense().getA()
        yte = yte.todense().getA()

        if dataset_name.endswith('-red'):
            TO_SELECT = 10
            nC = ytr.shape[1]
            tr_counts = ytr.sum(axis=0)
            te_counts = yte.sum(axis=0)
            if nC > TO_SELECT:
                Y = ytr.T.dot(ytr)  # class-class coincidence matrix
                Y[np.triu_indices(nC)] = 0  # zeroing all duplicates entries and the diagonal
                order_ij = np.argsort(-Y, axis=None)
                selected = set()
                p=0
                while len(selected) < TO_SELECT:
                    highest_index = order_ij[p]
                    class_i = highest_index // nC
                    class_j = highest_index % nC
                    # if there is nested_test_indexesay(sorted(selected))
                ytr = ytr[:,selected]
                yte = yte[:, selected]
        else:
            # remove categories without positives in the training
            valid_categories = ytr.sum(axis=0) >= 5
            ytr = ytr[:, valid_categories]
            yte = yte[:, valid_categories]

    elif dataset_name in TC_DATASETS:
        data = Dataset.load(dataset_name, pickle_path=f'{picklepath}/{dataset_name}.pickle')
        Xtr, Xte = data.vectorize()
        ytr = data.devel_labelmatrix.todense().getA()
        yte = data.test_labelmatrix.todense().getA()

        # remove categories with < 5 training documents
        to_keep = ytr.sum(axis=0) >= 5
        # keep the 10 most populated categories
        # to_keep = np.argsort(ytr.sum(axis=0))[-10:]
        ytr = ytr[:, to_keep]
        yte = yte[:, to_keep]
        print(f'num categories = {ytr.shape[1]}')

    else:
        raise ValueError(f'unknown dataset {dataset_name}')

    train = MultilabelledCollection(Xtr, ytr)
    test = MultilabelledCollection(Xte, yte)

    if dopickle:
        os.makedirs(datadir, exist_ok=True)
        pickle.dump((train, test), open(datapath, 'wb'), pickle.HIGHEST_PROTOCOL)

    return train, test


def already_run(result_path):
    if os.path.exists(result_path):
        print(f'{result_path} already computed. Skipping')
        return True
    return False


def print_info(train, test):
    # print((np.abs(np.corrcoef(ytr, rowvar=False))>0.1).sum())
    # sys.exit(0)

    print(f'Tr documents {len(train)}')
    print(f'Te documents {len(test)}')
    print(f'#features {train.instances.shape[1]}')
    print(f'#classes {train.labels.shape[1]}')

    # print(f'Train-prev: {train.prevalence()[:,1]}')
    print(f'Train-counts: {train.counts()}')
    # print(f'Test-prev: {test.prevalence()[:,1]}')
    print(f'Test-counts: {test.counts()}')
    print(f'MLPE: {qp.error.mae(train.prevalence(), test.prevalence()):.5f}')


# def save_results(npp_results, app_results, result_path, train_prevs):
#     # results are lists of tuples of (true_prevs, estim_prevs)
#     # each true_prevs is an ndarray of ndim=2, but the second dimension is constrained
#     def _prepare_result_lot(lot_results):
#         true_prevs, estim_prevs = lot_results
#         return {
#             'true_prevs': [true_i[:,0].flatten() for true_i in true_prevs],  # removes the constrained prevalence
#             'estim_prevs': [estim_i[:,0].flatten() for estim_i in estim_prevs],  # removes the constrained prevalence
#             'train_prevs' : train_prevs,
#         }
#     results = {
#         'npp': _prepare_result_lot(npp_results),
#         'app': _prepare_result_lot(app_results),
#     }
#     pickle.dump(results, open(result_path, 'wb'), pickle.HIGHEST_PROTOCOL)


# def load_results(result_path):
#     def _unpack_result_lot(lot_result):
#         true_prevs = lot_result['true_prevs']
#         true_prevs = [np.vstack([true_i, 1 - true_i]).T for true_i in true_prevs]  # add the constrained prevalence
#         estim_prevs = lot_result['estim_prevs']
#         estim_prevs = [np.vstack([estim_i, 1 - estim_i]).T for estim_i in estim_prevs]  # add the constrained prevalence
#         train_prevs = lot_result["train_prevs"]
#         return true_prevs, estim_prevs, train_prevs
#     results = pickle.load(open(result_path, 'rb'))
#     results = {
#         'npp': _unpack_result_lot(results['npp']),
#         'app': _unpack_result_lot(results['app']),
#     }
#     return results
    # results_npp = _unpack_result_lot(results['npp'])
    # results_app = _unpack_result_lot(results['app'])
    # return results_npp, results_app



def save_results(npp_results, app_results, result_path, train_prevs, best_params):
    # results are lists of tuples of (true_prevs, estim_prevs)
    # each true_prevs is an ndarray of ndim=2, but the second dimension is constrained
    def _prepare_result_lot(lot_results):
        true_prevs, estim_prevs = lot_results
        return {
            'true_prevs': np.asarray([true_i[:,1].flatten() for true_i in true_prevs]),  # removes the constrained prevalence
            'estim_prevs': np.asarray([estim_i[:,1].flatten() for estim_i in estim_prevs]),  # removes the constrained prevalence
            'train_prevs' : train_prevs,
            'best_params': best_params,
        }
    results = {
        'npp': _prepare_result_lot(npp_results),
        'app': _prepare_result_lot(app_results),
    }
    pickle.dump(results, open(result_path, 'wb'), pickle.HIGHEST_PROTOCOL)


def load_results(result_path):
    def _unpack_result_lot(lot_result):
        true_prevs = lot_result['true_prevs']
        true_prevs = [np.vstack([1-true_i, true_i]).T for true_i in true_prevs]  # add the constrained prevalence
        estim_prevs = lot_result['estim_prevs']
        estim_prevs = [np.vstack([1-estim_i, estim_i]).T for estim_i in estim_prevs]  # add the constrained prevalence
        train_prevs = lot_result["train_prevs"]

        best_params = None
        if "best_params" in lot_result.keys():
            best_params = lot_result["best_params"]

        return true_prevs, estim_prevs, train_prevs, best_params
    
    results = pickle.load(open(result_path, 'rb'))
    results = {
        'npp': _unpack_result_lot(results['npp']),
        'app': _unpack_result_lot(results['app']),
    }
    return results


def run_experiment(dataset_name, train, test, model_name, model, n_prevalences=101, repeats=25):
    result_path = f'{opt.results}/{dataset_name}_{model_name}.pkl'
    if already_run(result_path):
        return True

    print(f'runing experiment {dataset_name} x {model_name}')
    #train, test = get_dataset(dataset_name)

    model.fit(train)
    best_params = model.get_params()

    results_npp = ml_natural_prevalence_prediction(model, test, sample_size, repeats=n_prevalences*repeats)
    results_app = ml_artificial_prevalence_prediction(model, test, sample_size, n_prevalences=n_prevalences, repeats=repeats)

    save_results(results_npp, results_app, result_path, train.prevalence(), best_params)
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for multi-label quantification')
    parser.add_argument('--results', type=str, default='./results_generales', metavar='str',
                        help=f'path where to store the results')
    parser.add_argument('--subset', type=str, default="all", metavar="str",
                        help="subset of models to run, default: all, options: [general, mlc, mlq, all]")
    opt = parser.parse_args()

    os.makedirs(opt.results, exist_ok=True)


    # for datasetname, (modelname,model) in itertools.product(ALLTABLE[::-1], models()):
    #     run_experiment(datasetname, modelname, model)
    
    with open("whoami.txt", 'r') as f:
        who = f.readline().strip()
        if who == "all":
            dataset_list = itertools.chain(*SPLITS)
        elif who == "lla":
            dataset_list = reversed(list(itertools.chain(*SPLITS)))
        else:
            dataset_split = ["alex", "manolo", "amarna"].index(who)
            dataset_list = SPLITS[dataset_split]

    # import pandas as pd
    # records = []
    for dataset_name in dataset_list:
        train, test = get_dataset(dataset_name)

        # DEFAULTS
        n_prevalences = 101
        repeats = 1
        repeats_grid = 1
        n_prevalences_grid = 101
        if train.n_classes < 98:
            # DEFAULTS SMALL DATASETS
            n_prevalences = 101
            repeats = 25
            repeats_grid = 5
        elif train.n_classes > 500:
            # DEFAULTS HUGE DATASETS
            n_prevalences = 21
            n_prevalences_grid = 21
        
        handcrafted_repeats = {
            "jrcall": 1,
            "delicious": 1,
            "mediamill": 2,
            "birds": 40,
            "genbase": 50,
            "enron": 9,
            "ohsumed": 5,
            "bibtex": 2,
            "reuters21578":  6,
            "tmc2007_500": 5,
            "scene": 18,
            "medical": 15,
            "Corel5k": 6,
            "emotions": 26,
            "yeast": 8,
            "rcv1": 2,
        }

        if dataset_name in handcrafted_repeats.keys():
            repeats = handcrafted_repeats[dataset_name]
        else:
            print(f"There was not any particular number of repeats for dataset {dataset_name}, using defaults ({repeats}).")

        for modelname, model in models(opt.subset, n_prevalences=n_prevalences_grid, repeats=repeats_grid):
            if dataset_name == "delicious" and modelname == "CLEMS-PCC": #FIXME
                continue #FIXME
            
            # try:
            skipped = run_experiment(dataset_name, train, test, modelname, model, n_prevalences, repeats)

            # use the trained model to run its MRQ version
            if not skipped and not modelname.startswith("MRQ") and "CVStack" in modelname:
                common={'protocol':'app', 'sample_size':100, 'n_samples': 5000, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}
                
                modelname = f'MRQ-{modelname}'
                mrq_model = MLRegressionQuantification(model, **common)
                mrq_model.trained_ = True
                mrq_model.estimator_params_changed_ = False
                mrq_model.reg_params_changed_ = True
                gridq = MLGridSearchQ(
                    model=mrq_model,
                    param_grid={
                        'regressor__estimator__C': np.array([1., 10, 100, 1000, .1]),
                    },
                    sample_size=100,
                    n_jobs=-1,
                    verbose=True,
                    n_prevalences=n_prevalences,
                    repeats=repeats,
                )

                run_experiment(dataset_name, train, test, modelname, gridq, n_prevalences, repeats)

            # except Exception as e:
            #     print(f"Well there was some problem with {dataset_name} x {modelname}")
            #     print(e)

    

    #     stacked = np.vstack([train.labels, test.labels])
    #     combinations = pd.DataFrame(stacked).value_counts()
    #     ldiv = len(combinations)
    #     puniq = (combinations == 1).sum()
    #     pmax = combinations.iloc[0]
    #     records.append({
    #         "dataset": dataset_name,
    #         "train_size": train.instances.shape[0],
    #         "test_size": test.instances.shape[0],
    #         "n_labels": train.labels.shape[1],
    #         "cardinality": np.mean(np.sum(stacked, axis=1)),
    #         "density": np.mean(np.sum(stacked, axis=1)) / train.labels.shape[1],
    #         "diversity": ldiv,
    #         "normdiversity": ldiv / train.labels.shape[1],
    #         "puniq": puniq / train.instances.shape[0],
    #         "pmax": pmax / train.instances.shape[0],
    #     })
    #     continue
    
    # df = pd.DataFrame.from_records(records).sort_values("dataset")


#          dataset  train_size  test_size  n_labels  cardinality   density  diversity  normdiversity     puniq      pmax
# 0        Corel5k        4500        500       292     3.480000  0.011918       3113      10.660959  0.542889  0.012222
# 1         bibtex        4880       2515       159     2.401893  0.015106       2856      17.962264  0.450615  0.096516
# 2          birds         322        323        17     0.990698  0.058276        124       7.294118  0.204969  0.931677
# 3      delicious       12920       3185       983    19.019994  0.019349      15806      16.079349  1.210681  0.001471
# 4       emotions         391        202         6     1.868465  0.311411         27       4.500000  0.010230  0.207161
# 5          enron        1123        579        45     3.357227  0.074605        734      16.311111  0.490650  0.146928
# 6        genbase         463        199        18     1.219033  0.067724         23       1.277778  0.006479  0.369330
# 13        jrcall       13137       7233      1797     5.140501  0.002861      14560       8.102393  0.943823  0.019563
# 7      mediamill       30993      12914       100     4.374268  0.043743       6548      65.480000  0.132191  0.076243
# 8        medical         333        645        18     1.134969  0.063054         50       2.777778  0.033033  0.495495
# 14       ohsumed       24061      10328        23     1.657041  0.072045       1901      82.652174  0.041062  0.119530
# 15          rcv1       23149     781265        98     3.199423  0.032647      14820     151.224490  0.344983  2.323167
# 12  reuters21578        9603       3299        72     1.028988  0.014291        447       6.208333  0.027804  0.408518
# 9          scene        1211       1196         6     1.073951  0.178992         15       2.500000  0.002477  0.334434
# 10   tmc2007_500       21519       7077        22     2.219611  0.100891       1172      53.272727  0.018960  0.115433
# 11         yeast        1500        917        14     4.237071  0.302648        198      14.142857  0.051333  0.158000