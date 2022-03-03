import argparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import itertools

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from tqdm import tqdm
from skmultilearn.dataset import load_dataset, available_data_sets
from scipy.sparse import csr_matrix
import quapy as qp
from MultiLabel.mlclassification import MLStackedClassifier, LabelSpacePartion, MLTwinSVM, MLknn
from MultiLabel.mldata import MultilabelledCollection
from MultiLabel.mlquantification import MLNaiveQuantifier, MLCC, MLPCC, MLRegressionQuantification, \
    MLACC, \
    MLPACC, MLNaiveAggregativeQuantifier, MLMLPE, StackMLRQuantifier, MLadjustedCount, MLprobAdjustedCount
from quapy.method.aggregative import PACC, CC, EMQ, PCC, ACC, HDy
import numpy as np
from .quapy.data.dataset  import Dataset
from quapy.mlevaluation import ml_natural_prevalence_prediction, ml_artificial_prevalence_prediction
import sys
import os
import pickle


def cls():
    # return LinearSVC()
    return LogisticRegression(max_iter=1000, solver='lbfgs')


def calibratedCls():
    return CalibratedClassifierCV(cls())

# DEBUG=True

# if DEBUG:
sample_size = 100
n_samples = 5000

SKMULTILEARN_ALL_DATASETS = sorted(set([x[0] for x in available_data_sets().keys()]))
SKMULTILEARN_RED_DATASETS = [x+'-red' for x in SKMULTILEARN_ALL_DATASETS]
TC_DATASETS = ['reuters21578', 'jrcall', 'ohsumed', 'rcv1']

DATASETS = TC_DATASETS


def models():
    yield 'MLPE', MLMLPE()

    # naives (Binary Classification + Binary Quantification)
    yield 'NaiveCC', MLNaiveAggregativeQuantifier(CC(cls()))
    yield 'NaivePCC', MLNaiveAggregativeQuantifier(PCC(cls()))
    # yield 'NaivePCCcal', MLNaiveAggregativeQuantifier(PCC(calibratedCls()))
    yield 'NaiveACC', MLNaiveAggregativeQuantifier(ACC(cls()))
    yield 'NaivePACC', MLNaiveAggregativeQuantifier(PACC(cls()))
    # yield 'NaivePACCcal', MLNaiveAggregativeQuantifier(PACC(calibratedCls()))
    # yield 'NaiveACCit', MLNaiveAggregativeQuantifier(ACC(cls()))
    # yield 'NaivePACCit', MLNaiveAggregativeQuantifier(PACC(cls()))
    # yield 'NaiveHDy', MLNaiveAggregativeQuantifier(HDy(cls()))
    yield 'NaiveSLD', MLNaiveAggregativeQuantifier(EMQ(calibratedCls()))

    # Multi-label Classification + Binary Quantification
    yield 'StackCC', MLCC(MLStackedClassifier(cls()))
    # yield 'StackPCC', MLPCC(MLStackedClassifier(cls()))
    # yield 'StackPCCcal', MLPCC(MLStackedClassifier(calibratedCls()))
    # yield 'StackACC', MLACC(MLStackedClassifier(cls()))
    # yield 'StackPACC', MLPACC(MLStackedClassifier(cls()))
    # yield 'StackPACCcal', MLPACC(MLStackedClassifier(calibratedCls()))
    # yield 'StackACCit', MLACC(MLStackedClassifier(cls()))
    # yield 'StackPACCit', MLPACC(MLStackedClassifier(cls()))
    yield 'ChainCC', MLCC(ClassifierChain(cls(), cv=None))
    # yield 'ChainPCC', MLPCC(ClassifierChain(cls(), cv=None))
    # yield 'ChainACC', MLACC(ClassifierChain(cls(), cv=None))
    # yield 'ChainPACC', MLPACC(ClassifierChain(cls(), cv=None))

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
    common={'sample_size':sample_size, 'n_samples': n_samples, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}
    yield 'MRQ-CC', MLRegressionQuantification(MLNaiveQuantifier(CC(cls())), **common)
    # yield 'MRQ-PCC', MLRegressionQuantification(MLNaiveQuantifier(PCC(cls())), **common)
    # yield 'MRQ-ACC', MLRegressionQuantification(MLNaiveQuantifier(ACC(cls())), **common)
    # yield 'MRQ-PACC', MLRegressionQuantification(MLNaiveQuantifier(PACC(cls())), **common)
    # yield 'MRQ-ACCit', MLRegressionQuantification(MLNaiveQuantifier(ACC(cls())), **common)
    # yield 'MRQ-PACCit', MLRegressionQuantification(MLNaiveQuantifier(PACC(cls())), **common)

    # Multi-label Classification + Multi-label Quantification
    # yield 'MRQ-StackCC', MLRegressionQuantification(MLCC(MLStackedClassifier(cls())), **common)
    # yield 'MRQ-StackPCC', MLRegressionQuantification(MLPCC(MLStackedClassifier(cls())), **common)
    # yield 'MRQ-StackACC', MLRegressionQuantification(MLACC(MLStackedClassifier(cls())), **common)
    # yield 'MRQ-StackPACC', MLRegressionQuantification(MLPACC(MLStackedClassifier(cls())), **common)
    # yield 'MRQ-StackCC-app', MLRegressionQuantification(MLCC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-StackPCC-app', MLRegressionQuantification(MLPCC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-StackACC-app', MLRegressionQuantification(MLACC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-StackPACC-app', MLRegressionQuantification(MLPACC(MLStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-ChainCC', MLRegressionQuantification(MLCC(ClassifierChain(cls())), **common)
    # yield 'MRQ-ChainPCC', MLRegressionQuantification(MLPCC(ClassifierChain(cls())), **common)
    # yield 'MRQ-ChainACC', MLRegressionQuantification(MLACC(ClassifierChain(cls())), **common)
    # yield 'MRQ-ChainPACC', MLRegressionQuantification(MLPACC(ClassifierChain(cls())), **common)

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
        # else:
            # remove categories without positives in the training or test splits
            # valid_categories = np.logical_and(ytr.sum(axis=0)>5, yte.sum(axis=0)>5)
            # ytr = ytr[:, valid_categories]
            # yte = yte[:, valid_categories]

    elif dataset_name in TC_DATASETS:
        picklepath = '/home/manolo/Documentos/multi-label quantification/QuaPyPrivate/pickles_manuel'
        data = Dataset.load(dataset_name, pickle_path=f'{picklepath}/{dataset_name}.pickle')
        Xtr, Xte = data.vectorize()
        ytr = data.devel_labelmatrix.todense().getA()
        yte = data.test_labelmatrix.todense().getA()

        # remove categories with < 50 training or test documents
        # to_keep = np.logical_and(ytr.sum(axis=0)>=50, yte.sum(axis=0)>=50)
        # keep the 10 most populated categories
        to_keep = np.argsort(ytr.sum(axis=0))[-10:]
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


def save_results(npp_results, app_results, result_path):
    # results are lists of tuples of (true_prevs, estim_prevs)
    # each true_prevs is an ndarray of ndim=2, but the second dimension is constrained
    def _prepare_result_lot(lot_results):
        true_prevs, estim_prevs = lot_results
        return {
            'true_prevs': [true_i[:,0].flatten() for true_i in true_prevs],  # removes the constrained prevalence
            'estim_prevs': [estim_i[:,0].flatten() for estim_i in estim_prevs]  # removes the constrained prevalence
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
        return true_prevs, estim_prevs
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

    print_info(train, test)

    model.fit(train)

    results_npp = ml_natural_prevalence_prediction(model, test, sample_size, repeats=100)
    results_app = ml_artificial_prevalence_prediction(model, test, sample_size, n_prevalences=11, repeats=5)
    save_results(results_npp, results_app, result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for multi-label quantification')
    parser.add_argument('--results', type=str, default='./results', metavar='str',
                        help=f'path where to store the results')
    opt = parser.parse_args()

    os.makedirs(opt.results, exist_ok=True)

    for datasetname, (modelname,model) in itertools.product(DATASETS, models()):
        run_experiment(datasetname, modelname, model)





