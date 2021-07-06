from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from tqdm import tqdm

import quapy as qp
from MultiLabel.mlclassification import MultilabelStackedClassifier
from MultiLabel.mldata import MultilabelledCollection
from MultiLabel.mlquantification import MultilabelNaiveQuantifier, MLCC, MLPCC, MultilabelRegressionQuantification, \
    MLACC, \
    MLPACC, MultilabelNaiveAggregativeQuantifier
from method.aggregative import PACC, CC, EMQ, PCC, ACC, HDy
import numpy as np
from data.dataset  import Dataset
from mlevaluation import ml_natural_prevalence_evaluation, ml_artificial_prevalence_evaluation


def cls():
    # return LinearSVC()
    return LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)


def calibratedCls():
    return CalibratedClassifierCV(cls())

# DEBUG=True

# if DEBUG:
sample_size = 250
n_samples = 5000


def models():
    # yield 'NaiveCC', MultilabelNaiveAggregativeQuantifier(CC(cls()))
    # yield 'NaivePCC', MultilabelNaiveAggregativeQuantifier(PCC(cls()))
    # yield 'NaiveACC', MultilabelNaiveAggregativeQuantifier(ACC(cls()))
    # yield 'NaivePACC', MultilabelNaiveAggregativeQuantifier(PACC(cls()))
    # yield 'EMQ', MultilabelQuantifier(EMQ(calibratedCls()))
    # yield 'StackCC', MLCC(MultilabelStackedClassifier(cls()))
    # yield 'StackPCC', MLPCC(MultilabelStackedClassifier(cls()))
    # yield 'StackACC', MLACC(MultilabelStackedClassifier(cls()))
    # yield 'StackPACC', MLPACC(MultilabelStackedClassifier(cls()))
    # yield 'ChainCC', MLCC(ClassifierChain(cls(), cv=None, order='random'))
    # yield 'ChainPCC', MLPCC(ClassifierChain(cls(), cv=None, order='random'))
    # yield 'ChainACC', MLACC(ClassifierChain(cls(), cv=None, order='random'))
    # yield 'ChainPACC', MLPACC(ClassifierChain(cls(), cv=None, order='random'))
    common={'sample_size':sample_size, 'n_samples': n_samples, 'norm': True, 'means':False, 'stds':False}
    yield 'MRQ-CC', MultilabelRegressionQuantification(base_quantifier=CC(cls()), regression='svr', **common)
    yield 'MRQ-PCC', MultilabelRegressionQuantification(base_quantifier=PCC(cls()), regression='svr', **common)
    yield 'MRQ-ACC', MultilabelRegressionQuantification(base_quantifier=ACC(cls()), regression='svr', **common)
    yield 'MRQ-PACC', MultilabelRegressionQuantification(base_quantifier=PACC(cls()), regression='svr', **common)


dataset = 'reuters21578'
picklepath = '/home/moreo/word-class-embeddings/pickles'
data = Dataset.load(dataset, pickle_path=f'{picklepath}/{dataset}.pickle')

Xtr, Xte = data.vectorize()
ytr = data.devel_labelmatrix.todense().getA()
yte = data.test_labelmatrix.todense().getA()

# remove categories with < 10 training documents
to_keep = np.logical_and(ytr.sum(axis=0)>=50, yte.sum(axis=0)>=50)
ytr = ytr[:, to_keep]
yte = yte[:, to_keep]
print(f'num categories = {ytr.shape[1]}')

train = MultilabelledCollection(Xtr, ytr)
test = MultilabelledCollection(Xte, yte)

# print(f'Train-prev: {train.prevalence()[:,1]}')
print(f'Train-counts: {train.counts()}')
# print(f'Test-prev: {test.prevalence()[:,1]}')
print(f'Test-counts: {test.counts()}')
print(f'MLPE: {qp.error.mae(train.prevalence(), test.prevalence()):.5f}')

fit_models = {model_name:model.fit(train) for model_name,model in tqdm(models(), 'fitting', total=6)}

print('NPP:')
for model_name, model in fit_models.items():
    err = ml_natural_prevalence_evaluation(model, test, sample_size, repeats=100)
    print(f'{model_name:10s}\tmae={err:.5f}')

print('APP:')
for model_name, model in fit_models.items():
    err = ml_artificial_prevalence_evaluation(model, test, sample_size, n_prevalences=21, repeats=10)
    print(f'{model_name:10s}\tmae={err:.5f}')



