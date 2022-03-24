import signal
import traceback

signal.signal(signal.SIGUSR1, lambda sig, stack: traceback.print_stack(stack))



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


seed = 1
random.seed(seed)
np.random.seed(seed)


def cls():
    # return LinearSVC()
    return LogisticRegression(max_iter=2000, solver='lbfgs')


def calibratedCls():
    return CalibratedClassifierCV(cls())

sample_size = 5000 #MIRAR run_experiment
n_samples = 100 #MIRAR run_experiment

picklepath = 'pickles_manuel'

#SKMULTILEARN_ALL_DATASETS = sorted(set([x[0] for x in available_data_sets().keys()]))
SKMULTILEARN_ALL_DATASETS = ['Corel5k', 'bibtex', 'birds', 'delicious', 'emotions', 'enron', 'genbase', 'mediamill', 'medical', 'scene', 'tmc2007_500', 'yeast']
SKMULTILEARN_NOBIG_DATASETS = ['birds', 'emotions', 'enron', 'genbase', 'mediamill', 'medical', 'scene', 'tmc2007_500', 'yeast']
SKMULTILEARN_SMALL_DATASETS = ['birds', 'emotions', 'enron', 'genbase', 'medical', 'scene', 'tmc2007_500', 'yeast'] #offline
SKMULTILEARN_RED_DATASETS = [x+'-red' for x in SKMULTILEARN_ALL_DATASETS]
TC_DATASETS = ['reuters21578', 'jrcall', 'ohsumed', 'rcv1']
TC_DATASETS_REDUCED = ['rcv1', 'ohsumed']


SPLIT1 = ['birds', 'emotions', 'enron', 'reuters21578', 'jrcall'] #icx
SPLIT2 = ['genbase', 'mediamill', 'medical', 'Corel5k', 'bibtex',] #ilona
SPLIT3 = ['scene', 'tmc2007_500', 'yeast', 'ohsumed', 'rcv1', 'delicious'] #amarna


ALLTABLE = SKMULTILEARN_ALL_DATASETS + TC_DATASETS_REDUCED

#DATASETS = ['reuters21578']
COREL5K = ['Corel5k']

def select_best(model, param_grid=None, n_jobs=-1, single=False):
    if not param_grid:
        param_grid = dict(
            C=np.logspace(-3, 3, 7),
            class_weight=[None, "balanced"],
        )
    
    if single:
        return GridSearchQ(
            model=model,
            param_grid=param_grid,
            sample_size=100,
            n_jobs=1,
            verbose=False,
            n_prevpoints=21,
            protocol='app',
        )
    else:
        return MLGridSearchQ(
            model=model,
            param_grid=param_grid,
            sample_size=100,
            n_jobs=n_jobs,
            verbose=False,
        )


def models():
    common={'sample_size': sample_size, 'n_samples': n_samples, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}

    # naives (Binary Classification + Binary Quantification)
    yield 'NaiveCC', MLNaiveAggregativeQuantifier(select_best(CC(cls()), single=True))
    yield 'NaivePCC', MLNaiveAggregativeQuantifier(select_best(PCC(cls()), single=True))
    yield 'NaiveACC', MLNaiveAggregativeQuantifier(select_best(ACC(cls()), single=True))
    yield 'NaivePACC', MLNaiveAggregativeQuantifier(select_best(PACC(cls()), single=True))
    yield 'NaiveHDy', MLNaiveAggregativeQuantifier(select_best(HDy(cls()), single=True))
    yield 'NaiveSLDNoCalibrado', MLNaiveAggregativeQuantifier(select_best(EMQ(cls()), single=True))
    yield 'NaiveSLD', MLNaiveAggregativeQuantifier(select_best(EMQ(calibratedCls()), param_grid=dict(
            base_estimator__C=np.logspace(-3, 3, 7),
            base_estimator__class_weight=[None, "balanced"],
        ), single=True))

    # yield 'ss100-StackCC', select_best(MLCC(MLStackedClassifier(cls())))
    # yield 'ss100-StackPCC', select_best(MLPCC(MLStackedClassifier(cls())))
    # yield 'ss100-StackACC', select_best(MLACC(MLStackedClassifier(cls())))
    # yield 'ss100-StackPACC', select_best(MLPACC(MLStackedClassifier(cls())))

    common={'protocol':'app', 'sample_size':100, 'n_samples': 5000, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}
    # yield 'ss100-MRQ-CC', MLRegressionQuantification(MLNaiveQuantifier(select_best(CC(cls()), single=True)), **common)
    # yield 'ss100-MRQ-PCC', MLRegressionQuantification(MLNaiveQuantifier(select_best(PCC(cls()), single=True)), **common)
    # yield 'ss100-MRQ-ACC', MLRegressionQuantification(MLNaiveQuantifier(select_best(ACC(cls()), single=True)), **common)
    # yield 'ss100-MRQ-PACC', MLRegressionQuantification(MLNaiveQuantifier(select_best(PACC(cls()), single=True)), **common)

    # yield 'ss100-MRQ-StackCC', MLRegressionQuantification(select_best(MLCC(MLStackedClassifier(cls()))), **common)
    # yield 'ss100-MRQ-StackPCC', MLRegressionQuantification(select_best(MLPCC(MLStackedClassifier(cls()))), **common)
    # yield 'ss100-MRQ-StackACC', MLRegressionQuantification(select_best(MLACC(MLStackedClassifier(cls()))), **common)
    # yield 'ss100-MRQ-StackPACC', MLRegressionQuantification(select_best(MLPACC(MLStackedClassifier(cls()))), **common)




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
                    # if there is only one class to go, then add the most populated one
                    most_populated, least_populated = (class_i, class_j) if tr_counts[class_i] > tr_counts[class_j] else (class_j, class_i)
                    if te_counts[most_populated]>0:
                        selected.add(most_populated)
                    if len(selected) < TO_SELECT:
                        if te_counts[least_populated]>0:
                            selected.add(least_populated)
                    p+=1
                selected = np.asarray(sorted(selected))
                ytr = ytr[:,selected]
                yte = yte[:, selected]
        else:
            # remove categories without positives in the training or test splits
            valid_categories = np.logical_and(ytr.sum(axis=0)>5, yte.sum(axis=0)>5)
            ytr = ytr[:, valid_categories]
            yte = yte[:, valid_categories]

    elif dataset_name in TC_DATASETS:
        data = Dataset.load(dataset_name, pickle_path=f'{picklepath}/{dataset_name}.pickle')
        Xtr, Xte = data.vectorize()
        ytr = data.devel_labelmatrix.todense().getA()
        yte = data.test_labelmatrix.todense().getA()

        # remove categories with < 20 training or test documents
        to_keep = np.logical_and(ytr.sum(axis=0)>=5, yte.sum(axis=0)>=5)
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


def save_results(npp_results, app_results, result_path, train_prevs):
    # results are lists of tuples of (true_prevs, estim_prevs)
    # each true_prevs is an ndarray of ndim=2, but the second dimension is constrained
    def _prepare_result_lot(lot_results):
        true_prevs, estim_prevs = lot_results
        return {
            'true_prevs': [true_i[:,0].flatten() for true_i in true_prevs],  # removes the constrained prevalence
            'estim_prevs': [estim_i[:,0].flatten() for estim_i in estim_prevs],  # removes the constrained prevalence
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
        true_prevs = [np.vstack([true_i, 1 - true_i]).T for true_i in true_prevs]  # add the constrained prevalence
        estim_prevs = lot_result['estim_prevs']
        estim_prevs = [np.vstack([estim_i, 1 - estim_i]).T for estim_i in estim_prevs]  # add the constrained prevalence
        train_prevs = lot_result["train_prevs"]
        return true_prevs, estim_prevs, train_prevs
    results = pickle.load(open(result_path, 'rb'))
    results = {
        'npp': _unpack_result_lot(results['npp']),
        'app': _unpack_result_lot(results['app']),
    }
    return results
    # results_npp = _unpack_result_lot(results['npp'])
    # results_app = _unpack_result_lot(results['app'])
    # return results_npp, results_app


def run_experiment(dataset_name, model_name, model):
    result_path = f'{opt.results}/{dataset_name}_{model_name}.pkl'
    if already_run(result_path):
        return

    print(f'runing experiment {dataset_name} x {model_name}')
    train, test = get_dataset(dataset_name)
    # if train.n_classes>100:
    #     return

    model.fit(train)

    sample_size = 100

    results_npp = ml_natural_prevalence_prediction(model, test, sample_size, repeats=100)
    results_app = ml_artificial_prevalence_prediction(model, test, sample_size, n_prevalences=21, repeats=100)
    save_results(results_npp, results_app, result_path, train.prevalence())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for multi-label quantification')
    parser.add_argument('--results', type=str, default='./results_pruebas_COREL', metavar='str',
                        help=f'path where to store the results')
    opt = parser.parse_args()

    os.makedirs(opt.results, exist_ok=True)

    for datasetname, (modelname,model) in itertools.product(SKMULTILEARN_SMALL_DATASETS, models()):
        run_experiment(datasetname, modelname, model)





