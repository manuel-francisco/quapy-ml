from calendar import c
import signal
import traceback

signal.signal(signal.SIGUSR1, lambda sig, stack: traceback.print_stack(stack))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



import argparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, MultiTaskLasso
import itertools
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, RegressorChain
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from skmultilearn.dataset import load_dataset, available_data_sets
from scipy.sparse import csr_matrix
import quapy as qp
from MultiLabel.mlclassification import MLLabelClusterer, MLStackedClassifier, LabelSpacePartion, MLStackedRegressor, MLTwinSVM, MLknn, SKMLWrapper, MLEmbedding
from MultiLabel.mldata import MultilabelledCollection
from MultiLabel.mlquantification import MLCompositeCC, MLNaiveQuantifier, MLCC, MLPCC, MLRegressionQuantification, \
    MLACC, \
    MLPACC, MLNaiveAggregativeQuantifier, MLMLPE, MLSlicedCC, StackMLRQuantifier, MLadjustedCount, MLprobAdjustedCount, \
    CompositeMLRegressionQuantification, MLAggregativeQuantifier, MLQuantifier
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

from sklearn.linear_model import LogisticRegression, Ridge, Lasso, LassoCV, MultiTaskLassoCV, LassoLars, LassoLarsCV, \
    ElasticNet, MultiTaskElasticNetCV, MultiTaskElasticNet, LinearRegression, ARDRegression, BayesianRidge, SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


seed = 1
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


SPLITS = [
    ['rcv1', 'mediamill', 'ohsumed', 'tmc2007_500', 'reuters21578'],
    ['Corel5k', 'yeast', 'enron', 'delicious', 'jrcall'],
    ['medical', 'scene', 'emotions', 'birds', 'bibtex', 'genbase'],
]


ALLTABLE = SKMULTILEARN_ALL_DATASETS + TC_DATASETS

#DATASETS = ['reuters21578']
COREL5K = ['Corel5k']



def models(n_prevalences=101, repeats=25): # CAMBIAR EN __main__
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
    

    common={'sample_size': sample_size, 'n_samples': n_samples, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}

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

    yield 'StackCC', select_best(MLCC(MLStackedClassifier(cls())))
    yield 'StackPCC', select_best(MLPCC(MLStackedClassifier(cls())))
    yield 'StackACC', select_best(MLACC(MLStackedClassifier(cls())))
    yield 'StackPACC', select_best(MLPACC(MLStackedClassifier(cls())))

    common={'protocol':'app', 'sample_size':100, 'n_samples': 5000, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}
    yield 'MRQ-CC', MLRegressionQuantification(MLNaiveQuantifier(select_best(CC(cls()), single=True)), **common)
    yield 'MRQ-PCC', MLRegressionQuantification(MLNaiveQuantifier(select_best(PCC(cls()), single=True)), **common)
    yield 'MRQ-ACC', MLRegressionQuantification(MLNaiveQuantifier(select_best(ACC(cls()), single=True)), **common)
    yield 'MRQ-PACC', MLRegressionQuantification(MLNaiveQuantifier(select_best(PACC(cls()), single=True)), **common)
    yield 'MRQ-StackCC', MLRegressionQuantification(select_best(MLCC(MLStackedClassifier(cls()))), **common)
    yield 'MRQ-StackPCC', MLRegressionQuantification(select_best(MLPCC(MLStackedClassifier(cls()))), **common)
    yield 'MRQ-StackACC', MLRegressionQuantification(select_best(MLACC(MLStackedClassifier(cls()))), **common)
    yield 'MRQ-StackPACC', MLRegressionQuantification(select_best(MLPACC(MLStackedClassifier(cls()))), **common)




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
            # remove categories without positives in the training or test splits
            valid_categories = ytr.sum(axis=0) >= 5
            ytr = ytr[:, valid_categories]
            yte = yte[:, valid_categories]

    elif dataset_name in TC_DATASETS:
        data = Dataset.load(dataset_name, pickle_path=f'{picklepath}/{dataset_name}.pickle')
        Xtr, Xte = data.vectorize()
        ytr = data.devel_labelmatrix.todense().getA()
        yte = data.test_labelmatrix.todense().getA()

        # remove categories with < 20 training or test documents
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



def save_results(npp_results, app_results, result_path, train_prevs):
    # results are lists of tuples of (true_prevs, estim_prevs)
    # each true_prevs is an ndarray of ndim=2, but the second dimension is constrained
    def _prepare_result_lot(lot_results):
        true_prevs, estim_prevs = lot_results
        return {
            'true_prevs': np.asarray([true_i[:,1].flatten() for true_i in true_prevs]),  # removes the constrained prevalence
            'estim_prevs': np.asarray([estim_i[:,1].flatten() for estim_i in estim_prevs]),  # removes the constrained prevalence
            'train_prevs' : train_prevs,
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
        return true_prevs, estim_prevs, train_prevs
    results = pickle.load(open(result_path, 'rb'))
    results = {
        'npp': _unpack_result_lot(results['npp']),
        'app': _unpack_result_lot(results['app']),
    }
    return results


def run_experiment(dataset_name, train, test, model_name, model, n_prevalences=101, repeats=25):
    result_path = f'{opt.results}/{dataset_name}_{model_name}.pkl'
    if already_run(result_path):
        return

    print(f'runing experiment {dataset_name} x {model_name}')
    #train, test = get_dataset(dataset_name)

    model.fit(train)

    results_npp = ml_natural_prevalence_prediction(model, test, sample_size, repeats=n_prevalences*repeats)
    results_app = ml_artificial_prevalence_prediction(model, test, sample_size, n_prevalences=n_prevalences, repeats=repeats)
    save_results(results_npp, results_app, result_path, train.prevalence())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for multi-label quantification')
    parser.add_argument('--results', type=str, default='./results_generales', metavar='str',
                        help=f'path where to store the results')
    opt = parser.parse_args()

    os.makedirs(opt.results, exist_ok=True)


    # for datasetname, (modelname,model) in itertools.product(ALLTABLE[::-1], models()):
    #     run_experiment(datasetname, modelname, model)
    
    with open("whoami.txt", 'r') as f:
        who = f.readline().strip()
        dataset_split = ["alex", "manolo", "amarna"].index(who)

    for dataset_name in SPLITS[dataset_split]:
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
            print(f"EEEEEH debug: {dataset_name}")

        for modelname, model in models(n_prevalences=n_prevalences_grid, repeats=repeats_grid):
            run_experiment(dataset_name, train, test, modelname, model, n_prevalences, repeats)

    





class SSLR:
    def __init__(self, **params):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, solver='lbfgs', **params)
    
    def fit(self, X, y, **params):
        return self.clf.fit(self.scaler.fit_transform(X), y, **params)
    
    def predict(self, X, **params):
        return self.clf.predict(self.scaler.transform(X), **params)
    
    def predict_proba(self, X, **params):
        return self.clf.predict_proba(self.scaler.transform(X), **params)
    
    def get_params(self, deep=True):
        return self.clf.get_params(deep=deep)
    
    def set_params(self, **params):
        self.clf.set_params(**params)
    
    @property
    def classes_(self):
        return self.clf.classes_

# def cls():
#     # return LinearSVC()
#     return SSLR()