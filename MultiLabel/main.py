import argparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import itertools
from sklearn.multioutput import ClassifierChain
from tqdm import tqdm
from skmultilearn.dataset import load_dataset, available_data_sets
from scipy.sparse import csr_matrix
import quapy as qp
from MultiLabel.mlclassification import MultilabelStackedClassifier
from MultiLabel.mldata import MultilabelledCollection
from MultiLabel.mlquantification import MultilabelNaiveQuantifier, MLCC, MLPCC, MLRegressionQuantification, \
    MLACC, \
    MLPACC, MultilabelNaiveAggregativeQuantifier
from method.aggregative import PACC, CC, EMQ, PCC, ACC, HDy
import numpy as np
from data.dataset  import Dataset
from mlevaluation import ml_natural_prevalence_prediction, ml_artificial_prevalence_prediction
import sys
import os
import pickle


def cls():
    # return LinearSVC()
    return LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)


def calibratedCls():
    return CalibratedClassifierCV(cls())

# DEBUG=True

# if DEBUG:
sample_size = 100
n_samples = 5000


def models():
    yield 'NaiveCC', MultilabelNaiveAggregativeQuantifier(CC(cls()))
    yield 'NaivePCC', MultilabelNaiveAggregativeQuantifier(PCC(cls()))
    yield 'NaiveACC', MultilabelNaiveAggregativeQuantifier(ACC(cls()))
    yield 'NaivePACC', MultilabelNaiveAggregativeQuantifier(PACC(cls()))
    yield 'HDy', MultilabelNaiveAggregativeQuantifier(HDy(cls()))
    # yield 'EMQ', MultilabelQuantifier(EMQ(calibratedCls()))
    # yield 'StackCC', MLCC(MultilabelStackedClassifier(cls()))
    # yield 'StackPCC', MLPCC(MultilabelStackedClassifier(cls()))
    # yield 'StackACC', MLACC(MultilabelStackedClassifier(cls()))
    # yield 'StackPACC', MLPACC(MultilabelStackedClassifier(cls()))
    # yield 'ChainCC', MLCC(ClassifierChain(cls(), cv=None, order='random'))
    # yield 'ChainPCC', MLPCC(ClassifierChain(cls(), cv=None, order='random'))
    # yield 'ChainACC', MLACC(ClassifierChain(cls(), cv=None, order='random'))
    # yield 'ChainPACC', MLPACC(ClassifierChain(cls(), cv=None, order='random'))
    common={'sample_size':sample_size, 'n_samples': n_samples, 'norm': True, 'means':False, 'stds':False, 'regression':'svr'}
    # yield 'MRQ-CC', MLRegressionQuantification(MultilabelNaiveQuantifier(CC(cls())), **common)
    # yield 'MRQ-PCC', MLRegressionQuantification(MultilabelNaiveQuantifier(PCC(cls())),  **common)
    # yield 'MRQ-ACC', MLRegressionQuantification(MultilabelNaiveQuantifier(ACC(cls())),  **common)
    # yield 'MRQ-PACC', MLRegressionQuantification(MultilabelNaiveQuantifier(PACC(cls())), **common)
    # yield 'MRQ-StackCC', MLRegressionQuantification(MLCC(MultilabelStackedClassifier(cls())), **common)
    # yield 'MRQ-StackPCC', MLRegressionQuantification(MLPCC(MultilabelStackedClassifier(cls())), **common)
    # yield 'MRQ-StackACC', MLRegressionQuantification(MLACC(MultilabelStackedClassifier(cls())), **common)
    # yield 'MRQ-StackPACC', MLRegressionQuantification(MLPACC(MultilabelStackedClassifier(cls())),  **common)
    # yield 'MRQ-StackCC-app', MLRegressionQuantification(MLCC(MultilabelStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-StackPCC-app', MLRegressionQuantification(MLPCC(MultilabelStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-StackACC-app', MLRegressionQuantification(MLACC(MultilabelStackedClassifier(cls())), protocol='app', **common)
    # yield 'MRQ-StackPACC-app', MLRegressionQuantification(MLPACC(MultilabelStackedClassifier(cls())), protocol='app',  **common)
    # yield 'MRQ-ChainCC', MLRegressionQuantification(MLCC(ClassifierChain(cls())), **common)
    # yield 'MRQ-ChainPCC', MLRegressionQuantification(MLPCC(ClassifierChain(cls())), **common)
    # yield 'MRQ-ChainACC', MLRegressionQuantification(MLACC(ClassifierChain(cls())), **common)
    # yield 'MRQ-ChainPACC', MLRegressionQuantification(MLPACC(ClassifierChain(cls())), **common)


# dataset = 'reuters21578'
# picklepath = '/home/moreo/word-class-embeddings/pickles'
# data = Dataset.load(dataset, pickle_path=f'{picklepath}/{dataset}.pickle')
# Xtr, Xte = data.vectorize()
# ytr = data.devel_labelmatrix.todense().getA()
# yte = data.test_labelmatrix.todense().getA()

# remove categories with < 10 training documents
# to_keep = np.logical_and(ytr.sum(axis=0)>=50, yte.sum(axis=0)>=50)
# ytr = ytr[:, to_keep]
# yte = yte[:, to_keep]
# print(f'num categories = {ytr.shape[1]}')


def datasets():
    dataset_list = sorted(set([x[0] for x in available_data_sets().keys()]))
    for dataset_name in dataset_list:
        yield dataset_name


def get_dataset(dataset_name):
    Xtr, ytr, feature_names, label_names = load_dataset(dataset_name, 'train')
    Xte, yte, _, _ = load_dataset(dataset_name, 'test')
    print(f'n-labels = {len(label_names)}')

    Xtr = csr_matrix(Xtr)
    Xte = csr_matrix(Xte)

    ytr = ytr.todense().getA()
    yte = yte.todense().getA()

    # remove categories without positives in the training or test splits
    valid_categories = np.logical_and(ytr.sum(axis=0)>5, yte.sum(axis=0)>5)
    ytr = ytr[:, valid_categories]
    yte = yte[:, valid_categories]

    train = MultilabelledCollection(Xtr, ytr)
    test = MultilabelledCollection(Xte, yte)

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


def run_experiment(dataset_name, model_name, model):
    result_path = f'{opt.results}/{dataset_name}_{model_name}.pkl'
    if already_run(result_path):
        return

    print(f'runing experiment {dataset_name} x {model_name}')
    train, test = get_dataset(dataset_name)

    print_info(train, test)

    model.fit(train)

    results = dict()
    results['npp'] = ml_natural_prevalence_prediction(model, test, sample_size, repeats=100)
    results['app'] = ml_artificial_prevalence_prediction(model, test, sample_size, n_prevalences=21, repeats=10)
    pickle.dump(results, open(result_path, 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for multi-label quantification')
    parser.add_argument('--results', type=str, default='./results', metavar='str',
                        help=f'path where to store the results')
    opt = parser.parse_args()

    os.makedirs(opt.results, exist_ok=True)

    for datasetname, (modelname,model) in itertools.product(datasets(), models()):
        run_experiment(datasetname, modelname, model)



