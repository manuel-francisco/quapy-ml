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

# DEBUG=True

# if DEBUG:
sample_size = 100  # revise
n_samples = 5000

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
            verbose=True,
            n_prevpoints=21,
            protocol='app',
        )
    else:
        return MLGridSearchQ(
            model=model,
            param_grid=param_grid,
            sample_size=100,
            n_jobs=n_jobs,
            verbose=True,
        )


def models():
    common={'sample_size':sample_size, 'n_samples': n_samples, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}
    # yield 'MRQ-ChainPACC', MLRegressionQuantification(MLPACC(ClassifierChain(cls())), **common)

    #yield 'MLPE', MLMLPE()

    #yield 'PruebasCC', MLNaiveAggregativeQuantifier(CC(cls()))

    # naives (Binary Classification + Binary Quantification)
    yield 'NaiveCC', MLNaiveAggregativeQuantifier(select_best(CC(cls()), single=True))
    yield 'NaivePCC', MLNaiveAggregativeQuantifier(select_best(PCC(cls()), single=True))
    # yield 'NaivePCCcal', MLNaiveAggregativeQuantifier(PCC(calibratedCls()))
    yield 'NaiveACC', MLNaiveAggregativeQuantifier(select_best(ACC(cls()), single=True))
    yield 'NaivePACC', MLNaiveAggregativeQuantifier(select_best(PACC(cls()), single=True))
    # yield 'NaivePACCcal', MLNaiveAggregativeQuantifier(PACC(calibratedCls()))
    # yield 'NaiveACCit', MLNaiveAggregativeQuantifier(ACC(cls()))
    # yield 'NaivePACCit', MLNaiveAggregativeQuantifier(PACC(cls()))
    yield 'NaiveHDy', MLNaiveAggregativeQuantifier(select_best(HDy(cls()), single=True))
    yield 'NaiveSLD', MLNaiveAggregativeQuantifier(select_best(EMQ(calibratedCls()), single=True))
    yield 'NaiveSLDNoCalibrado', MLNaiveAggregativeQuantifier(select_best(EMQ(cls()), single=True))

    # Multi-label Classification + Binary Quantification
    yield 'StackCC', select_best(MLCC(MLStackedClassifier(cls())))
    yield 'StackPCC', select_best(MLPCC(MLStackedClassifier(cls())))
    # yield 'StackPCCcal', MLPCC(MLStackedClassifier(calibratedCls()))
    yield 'StackACC', select_best(MLACC(MLStackedClassifier(cls())))
    yield 'StackPACC', select_best(MLPACC(MLStackedClassifier(cls())))
    # yield 'StackPACCcal', MLPACC(MLStackedClassifier(calibratedCls()))
    # yield 'StackACCit', MLACC(MLStackedClassifier(cls()))
    # yield 'StackPACCit', MLPACC(MLStackedClassifier(cls()))
    # yield 'ChainCC', select_best(MLCC(ClassifierChain(cls())))
    # yield 'ChainPCC', select_best(MLPCC(ClassifierChain(cls())))
    # yield 'ChainACC', select_best(MLACC(ClassifierChain(cls())))
    # yield 'ChainPACC', select_best(MLPACC(ClassifierChain(cls())))



    #   -- Classifiers from scikit-multilearn
    # yield 'LSP-CC', MLCC(LabelSpacePartion(cls()))
    # yield 'LSP-ACC', MLACC(LabelSpacePartion(cls()))
    # yield 'TwinSVM-CC', MLCC(MLTwinSVM())
    # yield 'TwinSVM-ACC', MLACC(MLTwinSVM())
    # yield 'MLKNN-CC', MLCC(MLknn())
    # yield 'MLKNN-PCC', MLPCC(MLknn())
    # yield 'MLKNN-ACC', MLACC(MLknn())
    # yield 'MLKNN-PACC', MLPACC(MLknn())

    # Binary Classification + Multi-label Quantification
    common={'protocol':'app', 'sample_size':sample_size, 'n_samples': n_samples, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}
    yield 'MRQ-CC', MLRegressionQuantification(MLNaiveQuantifier(select_best(CC(cls()), single=True)), **common)
    yield 'MRQ-PCC', MLRegressionQuantification(MLNaiveQuantifier(select_best(PCC(cls()), single=True)), **common)
    yield 'MRQ-ACC', MLRegressionQuantification(MLNaiveQuantifier(select_best(ACC(cls()), single=True)), **common)
    yield 'MRQ-PACC', MLRegressionQuantification(MLNaiveQuantifier(select_best(PACC(cls()), single=True)), **common)
    # yield 'MRQ-ACCit', MLRegressionQuantification(MLNaiveQuantifier(ACC(cls())), alences=21,**common)
    # yield 'MRQ-PACCit', MLRegressionQuantification(MLNaiveQuantifier(PACC(cls())), **common)

    # Multi-label Classification + Multi-label Quantification
    yield 'MRQ-StackCC', MLRegressionQuantification(select_best(MLCC(MLStackedClassifier(cls()))), **common)
    yield 'MRQ-StackPCC', MLRegressionQuantification(select_best(MLPCC(MLStackedClassifier(cls()))), **common)
    yield 'MRQ-StackACC', MLRegressionQuantification(select_best(MLACC(MLStackedClassifier(cls()))), **common)
    yield 'MRQ-StackPACC', MLRegressionQuantification(select_best(MLPACC(MLStackedClassifier(cls()))), **common)
    # yield 'MRQ-StackCC-app', MLRegressionQuantification(MLCC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-StackPCC-app', MLRegressionQuantification(MLPCC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-StackACC-app', MLRegressionQuantification(MLACC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-StackPACC-app', MLRegressionQuantification(MLPACC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-ChainCC', MLRegressionQuantification(MLCC(ClassifierChain(cls())), **common)
    # yield 'MRQ-ChainPCC', MLRegressionQuantification(MLPCC(ClassifierChain(cls())), **common)
    # yield 'MRQ-ChainACC', MLRegressionQuantification(MLACC(ClassifierChain(cls())), **common)



    # MLC + BQ
    # yield 'CMRQ-CC', select_best(CompositeMLRegressionQuantification(MLNaiveQuantifier(CC(cls())), **common))
    # yield 'CMRQ-PCC', select_best(CompositeMLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), **common))
    # yield 'CMRQ-ACC', select_best(CompositeMLRegressionQuantification(MLNaiveQuantifier(ACC(cls())), **common))
    # yield 'CMRQ-PACC', select_best(CompositeMLRegressionQuantification(MLNaiveQuantifier(PACC(cls())), **common))
    # yield 'CMRQ-StackCC', select_best(CompositeMLRegressionQuantification(MLCC(MLStackedClassifier(cls())), MLCC(MLStackedClassifier(cls())), **common))
    # yield 'CMRQ-StackPCC', select_best(CompositeMLRegressionQuantification(MLPCC(MLStackedClassifier(cls())), MLPCC(MLStackedClassifier(cls())), **common))
    # yield 'CMRQ-StackACC', select_best(CompositeMLRegressionQuantification(MLACC(MLStackedClassifier(cls())), MLACC(MLStackedClassifier(cls())), **common))
    # yield 'CMRQ-StackPACC', select_best(CompositeMLRegressionQuantification(MLPACC(MLStackedClassifier(cls())), MLPACC(MLStackedClassifier(cls())), **common))
    # yield 'CMRQ-StackCC2', CompositeMLRegressionQuantification(MLCC(MLStackedClassifier(cls())), MLCC(MLStackedClassifier(cls())), k=2, **common)
    # yield 'CMRQ-StackPCC2', CompositeMLRegressionQuantification(MLPCC(MLStackedClassifier(cls())), MLPCC(MLStackedClassifier(cls())), k=2, **common)
    # yield 'CMRQ-StackACC2', CompositeMLRegressionQuantification(MLACC(MLStackedClassifier(cls())), MLACC(MLStackedClassifier(cls())), k=2, **common)
    # yield 'CMRQ-StackPACC2', CompositeMLRegressionQuantification(MLPACC(MLStackedClassifier(cls())), MLPACC(MLStackedClassifier(cls())), k=2, **common)
    # yield 'CMRQ-StackCC5', CompositeMLRegressionQuantification(MLCC(MLStackedClassifier(cls())), MLCC(MLStackedClassifier(cls())), k=5, **common)
    # yield 'CMRQ-StackPCC5', CompositeMLRegressionQuantification(MLPCC(MLStackedClassifier(cls())), MLPCC(MLStackedClassifier(cls())), k=5, **common)
    # yield 'CMRQ-StackACC5', CompositeMLRegressionQuantification(MLACC(MLStackedClassifier(cls())), MLACC(MLStackedClassifier(cls())), k=5, **common)
    # yield 'CMRQ-StackPACC5', CompositeMLRegressionQuantification(MLPACC(MLStackedClassifier(cls())), MLPACC(MLStackedClassifier(cls())), k=5, **common)


    # Chaos
    # yield 'StackMRQ-CC', StackMLRQuantifier(MLNaiveQuantifier(CC(cls())), **common)
    # yield 'StackMRQ-PCC', StackMLRQuantifier(MLNaiveQuantifier(PCC(cls())), **common)
    # yield 'StackMRQ-ACC', StackMLRQuantifier(MLNaiveQuantifier(ACC(cls())), **common)
    # yield 'StackMRQ-PACC', StackMLRQuantifier(MLNaiveQuantifier(PACC(cls())), **common)
    # yield 'StackMRQ-StackCC', StackMLRQuantifier(MLCC(MLStackedClassifier(cls())), **common)
    # yield 'StackMRQ-StackPCC', StackMLRQuantifier(MLPCC(MLStackedClassifier(cls())), **common)
    # yield 'StackMRQ-StackACC', StackMLRQuantifier(MLACC(MLStackedClassifier(cls())), **common)
    # yield 'StackMRQ-StackPACC', StackMLRQuantifier(MLPACC(MLStackedClassifier(cls())), **common)
    # yield 'StackMRQ-StackCC-app', StackMLRQuantifier(MLCC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'StackMRQ-StackPCC-app', StackMLRQuantifier(MLPCC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'StackMRQ-StackACC-app', StackMLRQuantifier(MLACC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'StackMRQ-StackPACC-app', StackMLRQuantifier(MLPACC(MLStackedClassifier(cls())), protocol='app', **common)

    # yield 'MLAdjustedC', MLadjustedCount(OneVsRestClassifier(cls()))
    # yield 'MLStackAdjustedC', MLadjustedCount(MLStackedClassifier(cls()))
    # yield 'MLprobAdjustedC', MLprobAdjustedCount(OneVsRestClassifier(calibratedCls()))
    # yield 'MLStackProbAdjustedC', MLprobAdjustedCount(MLStackedClassifier(calibratedCls()))



    # Multilabel classifiers exploration
    yield "MLkNN-MLPCC", select_best(MLPCC(SKMLWrapper(MLkNN())), param_grid={'k': range(1,10,2), 's': [0.5, 0.7, 1.0]})
    yield 'ChainPCC', select_best(MLPCC(ClassifierChain(cls())), param_grid={
        'base_estimator__C': np.logspace(-3, 3, 7),
        'base_estimator__class_weight': [None, "balanced"],
    })
    yield 'CLEMS-PCC', select_best(MLPCC(MLEmbedding()), param_grid={
        'regressor__n_estimators': [10, 20, 50],
        'classifier__k': range(1, 10, 2),
        'classifier__s': [.5, .7, 1.],
    }, n_jobs=1)
    #np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    yield 'LClusterer-PCC', select_best(MLPCC(MLLabelClusterer()), param_grid={
        # 'classifier__k': range(1,10,2),
        # 'classifier__s': [0.5, 0.7, 1.0],
        'clusterer__n_clusters': [2,3,5],
    }, n_jobs=6)

    yield 'DT-PCC', select_best(MLPCC(DecisionTreeClassifier()), param_grid={
        'criterion': ["gini", "entropy"],
        #'classifier__class_weight': [None, "balanced"],
    })

    yield 'RF-PCC', select_best(MLPCC(RandomForestClassifier(n_jobs=6)), param_grid={
        'n_estimators': [10, 100, 200],
        #'classifier__criterion': ["gini", "entropy"],
    }, n_jobs=1)

    # yield 'RadiusNeighbours-PCC', select_best(MLPCC(RadiusNeighborsClassifier(n_jobs=-1)), param_grid={
    #     'classifier__weights': ["uniform", "distance"],
    #     'classifier__radius': [.5, .8, 1., 1.5],
    #     'classifier_p': [1, 2],
    #     'classifier__criterion': ["gini", "entropy"],
    # })

    # yield 'MLPClassifier-PCC', select_best(MLPCC(MLPClassifier()), param_grid={
    #     'classifier__weights': ["uniform", "distance"],
    #     'classifier__radius': [.5, .8, 1., 1.5],
    #     'classifier_p': [1, 2],
    #     'classifier__criterion': ["gini", "entropy"],
    # })
    # CLEMS
    # yield "MLkNN-MLPCC", MLPCC(SKMLWrapper(MLkNN()))
    # yield "MLkNN-MLACC", MLACC(MLkNN())
    # yield "MLkNN-MLPACC", MLPACC(MLkNN())
    # yield "MLARAM-MLCC", MLCC(SKMLWrapper(MLARAM()))
    # yield "MLARAM-MLPCC", MLPCC(SKMLWrapper(MLARAM()))
    # yield "MLARAM-MLACC", MLACC(MLARAM())
    # yield "MLARAM-MLPACC", MLPACC(MLARAM())
    # yield "MLTSVM-MLCC", MLCC(SKMLWrapper(MLTSVM()))
    # yield "MLTSVM-MLPCC", MLPCC(SKMLWrapper(MLTSVM()))
    # yield "MLTSVM-MLACC", MLACC(MLTSVM())
    # yield "MLTSVM-MLPACC", MLPACC(MLTSVM())

    common={'protocol':'app', 'sample_size':sample_size, 'n_samples': n_samples, 'norm': True, 'means':False, 'stds':False}
    yield 'PCC-MRQ-MultitaskLasso', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=MultiTaskLasso(normalize=True), **common), param_grid={
        'alpha': np.linspace(0.001, 0.03, 5),
    })
    yield 'PCC-MRQ-Ridge', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=Ridge(normalize=True), **common), param_grid={
        'alpha': [200, 235, 270, 300, 500],
    })
    yield 'PCC-MRQ-LinearRegression', MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=LinearRegression(n_jobs=-1), **common)
    yield 'PCC-MRQ-RandomForest', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=RandomForestRegressor(n_jobs=6), **common), param_grid={
        "n_estimators": [10, 100, 200],
    }, n_jobs=1)
    yield 'PCC-AdaBoostChain', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=RegressorChain(AdaBoostRegressor()), **common), param_grid={
        'n_estimators': [10, 50, 100, 200]
    })
    yield 'PCC-AdaBoostStack', select_best(MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), regression=MLStackedRegressor(AdaBoostRegressor()), **common), param_grid={
        'reg__n_estimators': [10, 50, 100, 200]
    })




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

    results_npp = ml_natural_prevalence_prediction(model, test, sample_size, repeats=100)
    results_app = ml_artificial_prevalence_prediction(model, test, sample_size, n_prevalences=11, repeats=5)
    save_results(results_npp, results_app, result_path, train.prevalence())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for multi-label quantification')
    parser.add_argument('--results', type=str, default='./results_optimised', metavar='str',
                        help=f'path where to store the results')
    opt = parser.parse_args()

    os.makedirs(opt.results, exist_ok=True)

    for datasetname, (modelname,model) in itertools.product(SPLIT1, models()):
        run_experiment(datasetname, modelname, model)





