import argparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import itertools
from sklearn.multioutput import ClassifierChain
from tqdm import tqdm
from skmultilearn.dataset import load_dataset, available_data_sets
from scipy.sparse import csr_matrix
import quapy as qp
from MultiLabel.main import load_results, SKMULTILEARN_RED_DATASETS, TC_DATASETS, sample_size
from MultiLabel.mlclassification import MLStackedClassifier
from MultiLabel.mldata import MultilabelledCollection
from MultiLabel.mlquantification import MLNaiveQuantifier, MLCC, MLPCC, MLRegressionQuantification, \
    MLACC, \
    MLPACC, MLNaiveAggregativeQuantifier
from MultiLabel.tabular import Table
from quapy.method.aggregative import PACC, CC, EMQ, PCC, ACC, HDy
import numpy as np
from data.dataset  import Dataset
from mlevaluation import ml_natural_prevalence_prediction, ml_artificial_prevalence_prediction, check_error_str
import sys
import os
import pickle

models = ['MLPE',
    'NaiveCC', 'NaivePCC', 'NaiveACC', 'NaivePACC', #'NaivePACCcal', 'NaiveACCit', 'NaivePACCit',
    #'NaiveHDy', 'NaiveSLD',
    'ChainCC', 'ChainPCC', 'ChainACC', 'ChainPACC',
    'StackCC', 'StackPCC', #'StackPCCcal',
    'StackACC', 'StackPACC', #'StackPACCcal', 'StackACCit', 'StackP'
    #                                                               'ACCit',
    #'CompositeCC', 'SlicedCC', 'CMRQ-CC', 'CMRQ-StackCC',
    'MRQ-CC', 'MRQ-PCC', 'MRQ-ACC', 'MRQ-PACC', # 'MRQ-ACCit', 'MRQ-PACCit',
    # 'StackMRQ-CC', 'StackMRQ-PCC', 'StackMRQ-ACC', 'StackMRQ-PACC',
    'MRQ-StackCC', 'MRQ-StackPCC', 'MRQ-StackACC', 'MRQ-StackPACC',
    'MRQ-ChainCC', 'MRQ-ChainPCC', 'MRQ-ChainACC', 'MRQ-ChainPACC',
    'CMRQ-CC', 'CMRQ-PCC', 'CMRQ-ACC', 'CMRQ-PACC',
    'CMRQ-StackCC', 'CMRQ-StackPCC', 'CMRQ-StackACC', 'CMRQ-StackPACC',
    'CMRQ-StackCC2', 'CMRQ-StackPCC2', 'CMRQ-StackACC2', 'CMRQ-StackPACC2',
    'CMRQ-StackCC5', 'CMRQ-StackPCC5', 'CMRQ-StackACC5', 'CMRQ-StackPACC5',
    # 'StackMRQ-StackCC', 'StackMRQ-StackPCC', 'StackMRQ-StackACC', 'StackMRQ-StackPACC',
    # 'MRQ-StackCC-app', 'MRQ-StackPCC-app', 'MRQ-StackACC-app', 'MRQ-StackPACC-app',
    # 'StackMRQ-StackCC-app', 'StackMRQ-StackPCC-app', 'StackMRQ-StackACC-app', 'StackMRQ-StackPACC-app',
    # 'LSP-CC', 'LSP-ACC', 'MLKNN-CC', 'MLKNN-ACC',
    # 'MLAdjustedC', 'MLStackAdjustedC', 'MLprobAdjustedC', 'MLStackProbAdjustedC'
]

# datasets = sorted(set([x[0] for x in available_data_sets().keys()]))
datasets = TC_DATASETS
datasets.extend(['Corel5k', 'bibtex', 'birds', 'delicious', 'emotions', 'enron', 'genbase', 'mediamill', 'medical', 'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'rcv1subset5', 'scene', 'tmc2007_500', 'yeast'])



def generate_table(path, protocol, error):

    def compute_score_job(args):
        dataset, model = args
        result_path = f'{opt.results}/{dataset}_{model}.pkl'
        if os.path.exists(result_path):
            print('+', end='')
            sys.stdout.flush()
            result = load_results(result_path)
            true_prevs, estim_prevs = result[protocol]
            scores = np.asarray([error(trues, estims) for trues, estims in zip(true_prevs, estim_prevs)]).flatten()
            return dataset, model, scores
        print('-', end='')
        sys.stdout.flush()
        return None


    print(f'\ngenerating {path}')
    table = Table(datasets, models, prec_mean=4, significance_test='wilcoxon')
    results = qp.util.parallel(compute_score_job, list(itertools.product(datasets, models)), n_jobs=-1)
    print()

    for r in results:
        if r is not None:
            dataset, model, scores = r
            table.add(dataset, model, scores)

    save_table(table, path)
    save_table(table.getRankTable(), path.replace('.tex', '.rank.tex'))



def save_table(table, path):
    tabular = """
    \\resizebox{\\textwidth}{!}{%
            \\begin{tabular}{|c||""" + ('c|' * (len(datasets)+1)) + """} \hline
            """
    dataset_replace = {'tmc2007_500': 'tmc2007\_500', 'tmc2007_500-red': 'tmc2007\_500-red'}
    method_replace = {}

    tabular += table.latexTabularT(benchmark_replace=dataset_replace, method_replace=method_replace, side=True)
    tabular += """
        \end{tabular}%
        }
    """
    with open(path, 'wt') as foo:
        foo.write(tabular)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for multi-label quantification')
    parser.add_argument('--results', type=str, default='./results', metavar='str',
                        help=f'path where to store the results')
    parser.add_argument('--tablepath', type=str, default='./tables', metavar='str',
                        help=f'path where to store the tables')
    opt = parser.parse_args()

    assert os.path.exists(opt.results), f'result directory {opt.results} does not exist'
    os.makedirs(opt.tablepath, exist_ok=True)

    qp.environ["SAMPLE_SIZE"] = sample_size
    absolute_error = qp.error.ae
    relative_absolute_error = qp.error.rae

    generate_table(f'{opt.tablepath}/npp.ae.tex', protocol='npp', error=absolute_error)
    generate_table(f'{opt.tablepath}/app.ae.tex', protocol='app', error=absolute_error)
    generate_table(f'{opt.tablepath}/npp.rae.tex', protocol='npp', error=relative_absolute_error)
    generate_table(f'{opt.tablepath}/app.rae.tex', protocol='app', error=relative_absolute_error)





