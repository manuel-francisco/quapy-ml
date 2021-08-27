import argparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import itertools
from sklearn.multioutput import ClassifierChain
from tqdm import tqdm
from skmultilearn.dataset import load_dataset, available_data_sets
from scipy.sparse import csr_matrix
import quapy as qp
from MultiLabel.main import load_results
from MultiLabel.mlclassification import MultilabelStackedClassifier
from MultiLabel.mldata import MultilabelledCollection
from MultiLabel.mlquantification import MultilabelNaiveQuantifier, MLCC, MLPCC, MLRegressionQuantification, \
    MLACC, \
    MLPACC, MultilabelNaiveAggregativeQuantifier
from MultiLabel.tabular import Table
from method.aggregative import PACC, CC, EMQ, PCC, ACC, HDy
import numpy as np
from data.dataset  import Dataset
from mlevaluation import ml_natural_prevalence_prediction, ml_artificial_prevalence_prediction, check_error_str
import sys
import os
import pickle

models = ['NaiveCC', 'NaivePCC', 'NaiveACC', 'NaivePACC', 'NaiveHDy', 'NaiveSLD']
datasets = sorted(set([x[0] for x in available_data_sets().keys()]))


def generate_table(path, protocol, error):
    print(f'generating {path}')
    table = Table(datasets, models)
    for dataset, model in itertools.product(datasets, models):
        result_path = f'{opt.results}/{dataset}_{model}.pkl'
        if os.path.exists(result_path):
            result = load_results(result_path)
            true_prevs, estim_prevs = result[protocol]
            scores = np.asarray([error(trues, estims) for trues, estims in zip(true_prevs, estim_prevs)]).flatten()
            table.add(dataset, model, scores)

    tabular = """
    \\resizebox{\\textwidth}{!}{%
            \\begin{tabular}{|c||""" + ('c|' * len(models)) + """} \hline
            """
    dataset_replace = {'tmc2007_500': 'tmc2007\_500'}
    method_replace = {}

    tabular += table.latexTabular(benchmark_replace=dataset_replace, method_replace=method_replace)
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

    os.makedirs(opt.results, exist_ok=True)
    os.makedirs(opt.tablepath, exist_ok=True)

    eval_error = qp.error.ae
    generate_table(f'{opt.tablepath}/npp.ae.tex', protocol='npp', error=eval_error)
    generate_table(f'{opt.tablepath}/app.ae.tex', protocol='app', error=eval_error)






