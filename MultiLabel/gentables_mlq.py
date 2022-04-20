import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import itertools
from sklearn.multioutput import ClassifierChain
from tqdm import tqdm
from skmultilearn.dataset import load_dataset, available_data_sets
from scipy.sparse import csr_matrix
import quapy as qp
from MultiLabel.main_pruebas import load_results, SKMULTILEARN_RED_DATASETS, TC_DATASETS, sample_size
from MultiLabel.mlclassification import MLStackedClassifier
from MultiLabel.mldata import MultilabelledCollection
from MultiLabel.mlquantification import MLNaiveQuantifier, MLCC, MLPCC, MLRegressionQuantification, \
    MLACC, \
    MLPACC, MLNaiveAggregativeQuantifier
from MultiLabel.tabular import Table, MultiMethodTable
from quapy.method.aggregative import PACC, CC, EMQ, PCC, ACC, HDy
import numpy as np
from data.dataset  import Dataset
from mlevaluation import ml_natural_prevalence_prediction, ml_artificial_prevalence_prediction, check_error_str
import sys
import os
import pickle

models = [
    #'MLPE',
    'NaiveCC', 'NaivePCC', 'NaiveACC', 'NaivePACC', #'NaivePACCcal', 'NaiveACCit', 'NaivePACCit',
    # 'NaiveHDy', 'NaiveSLD',
    # 'ChainCC', 'ChainPCC', 'ChainACC', 'ChainPACC',
    'StackCC', 'StackPCC', #'StackPCCcal',
    'StackACC', 'StackPACC', #'StackPACCcal', 'StackACCit', 'StackP'
    #                                                               'ACCit',
    #'CompositeCC', 'SlicedCC', 'CMRQ-CC', 'CMRQ-StackCC',
    'MRQ-CC', 'MRQ-PCC', 'MRQ-ACC', 'MRQ-PACC', # 'MRQ-ACCit', 'MRQ-PACCit',
    # 'StackMRQ-CC', 'StackMRQ-PCC', 'StackMRQ-ACC', 'StackMRQ-PACC',
    'MRQ-StackCC', 'MRQ-StackPCC', 'MRQ-StackACC', 'MRQ-StackPACC',
    # 'MRQ-ChainCC', 'MRQ-ChainPCC', 'MRQ-ChainACC', 'MRQ-ChainPACC',
    # 'CMRQ-CC', 'CMRQ-PCC', 'CMRQ-ACC', 'CMRQ-PACC',
    # 'CMRQ-StackCC', 'CMRQ-StackPCC', 'CMRQ-StackACC', 'CMRQ-StackPACC',
    # 'CMRQ-StackCC2', 'CMRQ-StackPCC2', 'CMRQ-StackACC2', 'CMRQ-StackPACC2',
    # 'CMRQ-StackCC5', 'CMRQ-StackPCC5', 'CMRQ-StackACC5', 'CMRQ-StackPACC5',
    # 'StackMRQ-StackCC', 'StackMRQ-StackPCC', 'StackMRQ-StackACC', 'StackMRQ-StackPACC',
    # 'MRQ-StackCC-app', 'MRQ-StackPCC-app', 'MRQ-StackACC-app', 'MRQ-StackPACC-app',
    # 'StackMRQ-StackCC-app', 'StackMRQ-StackPCC-app', 'StackMRQ-StackACC-app', 'StackMRQ-StackPACC-app',
    # 'LSP-CC', 'LSP-ACC', 'MLKNN-CC', 'MLKNN-ACC',
    # 'MLAdjustedC', 'MLStackAdjustedC', 'MLprobAdjustedC', 'MLStackProbAdjustedC'
    # 'MLkNN-MLCC', 'MLkNN-MLPCC', 'MLkNN-MLACC', 'MLkNN-MLPACC',
    # 'MLARAM-MLCC', 'MLARAM-MLPCC', 'MLARAM-MLACC', 'MLARAM-MLPACC',
    # 'MLTSVM-MLCC', 'MLTSVM-MLACC',
    # 'MRQ-MultitaskLassoCV',
    # 'MRQ-Ridge',
    # 'MRQ-MultiTaskElasticNetCV',
    # 'MRQ-LinearRegression',
    # 'MRQ-DecisionTree',
    # 'MRQ-RandomForest',
    # 'MRQ-GaussianProcess',
    # 'MRQ-RANSAC',
    # 'MRQ-kNNReg',
    # # 'MRQ-RadiusNeigh',
    # 'MRQ-MLPReg',
    # 'MRQ-ChainLinearSVR',
    # 'MRQ-MOLassoLarsCV',
    # 'MRQ-MOARDRegression',
    # 'MRQ-MOBayesianRidge',
    # 'MRQ-MOLinearSVR',

    # 'NaivePCC',
    # 'StackPCC',
    # 'PCC-MRQ-MultitaskLassoCV',
    # 'PCC-MRQ-Ridge',
    # 'PCC-MRQ-MultiTaskElasticNetCV',
    # 'PCC-MRQ-LinearRegression',
    # 'PCC-MRQ-DecisionTree',
    # 'PCC-MRQ-RandomForest',
    # 'PCC-MRQ-GaussianProcess',
    # 'PCC-MRQ-RANSAC',
    # 'PCC-MRQ-kNNReg',
    # 'PCC-MRQ-MLPReg',
    # 'PCC-MRQ-ChainLinearSVR',
    # 'PCC-MRQ-MOLassoLarsCV',
    # 'PCC-MRQ-MOARDRegression',
    # 'PCC-MRQ-MOBayesianRidge',
    # 'PCC-MRQ-MOLinearSVR',
]


model_subsets = {
    "general": [
        'NaiveCC', 'NaivePCC', 'NaiveACC', 'NaivePACC',
        'CVStackCC', 'CVStackPCC', 'CVStackACC', 'CVStackPACC',
        'MRQ-CC', 'MRQ-PCC', 'MRQ-ACC', 'MRQ-PACC',
        'MRQ-CVStackCC', 'MRQ-CVStackPCC', 'MRQ-CVStackACC', 'MRQ-CVStackPACC',
    ],
    "mlc": [
        'MLkNN-MLPCC', 'ChainPCC', 'CLEMS-PCC', 'LClusterer-PCC', 'DT-PCC', 'RF-PCC',
    ],
    "mlq": [
        'RakelD-PCC', 'KMeansClustersPowerSet-PCC', 'MRQ-PCC',
    ],
    "mlq-reg": [
        'MRQ-Ridge', 'MRQ-MultitaskLasso', 'MRQ-RandomForest',
        'MRQ-LinearSVR', 'MRQ-StackedLinearSVR', 'MRQ-ChainedLinearSVR',
    ]
}



# datasets = sorted(set([x[0] for x in available_data_sets().keys()]))
TC_DATASETS = ['reuters21578', 'jrcall', 'ohsumed', 'rcv1']
#datasets = ['Corel5k', 'bibtex', 'birds', 'delicious', 'emotions', 'enron', 'genbase', 'mediamill', 'medical', 'scene', 'tmc2007_500', 'yeast', 'rcv1', 'ohsumed']
SKMULTILEARN_ALL_DATASETS = ['Corel5k', 'bibtex', 'birds', 'delicious', 'emotions', 'enron', 'genbase', 'mediamill', 'medical', 'scene', 'tmc2007_500', 'yeast']
SKMULTILEARN_NOBIG_DATASETS = ['birds', 'emotions', 'enron', 'genbase', 'mediamill', 'medical', 'scene', 'tmc2007_500', 'yeast']
SKMULTILEARN_SMALL_DATASETS = ['birds', 'emotions', 'enron', 'genbase', 'medical', 'scene', 'tmc2007_500', 'yeast'] #offline
ALLTABLE = SKMULTILEARN_ALL_DATASETS + TC_DATASETS
datasets = ALLTABLE


def generate_table(path, protocol, error, include, color=True, prec_mean=4):#, drift_bin):

    def compute_score_job(dataset, model):
        if include == "all" or any(model.endswith(e) for e in include):
            result_path = f'{opt.results}/{dataset}_{model}.pkl'
            if os.path.exists(result_path):
                print('+', end='')
                sys.stdout.flush()
                result = load_results(result_path)
                true_prevs = result[protocol][0]
                estim_prevs = result[protocol][1]
                train_prevs = result[protocol][2]
                train_prevs = np.asarray(train_prevs)
                scores = np.asarray([error(trues, estims) for trues, estims in zip(true_prevs, estim_prevs)])

                drifts = np.asarray([abs(trues[:, 1]-train_prevs[:, 1]).sum() for trues in true_prevs])
                # drifts = np.mean(drifts, axis=1)
                min_drifts = np.min(drifts)
                max_drifts = np.max(drifts)
                delta_drifts = (max_drifts - min_drifts) / 3
                bins = [0]
                bins.extend((min_drifts + (drift_bin + 1) * delta_drifts for drift_bin in range(3)))
                bins.append(np.max(drifts)+1)
                assigned_bin = np.digitize(drifts, bins, right=True) - 1

                for drift_bin in range(3):
                    which = np.flatnonzero(assigned_bin == drift_bin)
                    s_aux = scores[which, :].flatten()

                    yield dataset, model, drift_bin, s_aux
                return None
            print('-', end='')
            sys.stdout.flush()
            return None
        return None


    print(f'\ngenerating {path}')
    filtered_models = [m for m in models if include == "all" or any(m.endswith(e) for e in include)]
    table = MultiMethodTable(sorted(datasets), filtered_models, prec_mean=prec_mean, clean_zero=True, significance_test=None, color=color) # FIXME significance_test='wilcoxon'
    # results = qp.util.parallel(compute_score_job, list(itertools.product(datasets, filtered_models)), n_jobs=1) #FIXME n_jobs=-1
    print()

    # for r in results:
    for args in itertools.product(datasets, filtered_models):
        # r = compute_score_job(*args)
        # if r is not None:
        #     dataset, model, scores = r
        #     table.add(dataset, model, scores)
        for r in compute_score_job(*args):
            dataset, model, drift_bin, scores = r
            table.add(dataset, model, drift_bin, scores)

    save_table(table, path, filtered_models)
    # save_table(table.getRankTable(), path.replace('.tex', '.rank.tex'), filtered_models)


def save_table(table, path, filtered_models):
    tabular = """
    \\resizebox{\\textwidth}{!}{%
            \\begin{tabularx}{\\textwidth}{@{}l""" + ('Y' * (len(list(itertools.product(filtered_models, range(3))))+1)) + """@{}} \\toprule
            """
    dataset_replace = {'tmc2007_500': 'tmc2007\_500', 'tmc2007_500-red': 'tmc2007\_500-red',
    'tmc2007_5000': 'tmc2007\_5000',
    'tmc2007_5001': 'tmc2007\_5001',
    'tmc2007_5002': 'tmc2007\_5002',}
    method_replace = {
        "NaiveCC": "\BC\SEP\BQ",
        "StackCC": "\MLC\SEP\BQ",
        "MRQ-CC": "\BC\SEP\MLQ",
        "MRQ-StackCC": "\MLC\SEP\MLQ",
        "NaivePCC": "\BC\SEP\BQ",
        "StackPCC": "\MLC\SEP\BQ",
        "MRQ-PCC": "\BC\SEP\MLQ",
        "MRQ-StackPCC": "\MLC\SEP\MLQ",
        "NaiveACC": "\BC\SEP\BQ",
        "StackACC": "\MLC\SEP\BQ",
        "MRQ-ACC": "\BC\SEP\MLQ",
        "MRQ-StackACC": "\MLC\SEP\MLQ",
        "NaivePACC": "\BC\SEP\BQ",
        "StackPACC": "\MLC\SEP\BQ",
        "MRQ-PACC": "\BC\SEP\MLQ",
        "MRQ-StackPACC": "\MLC\SEP\MLQ",

        #MLC
        'MLkNN-MLPCC': '1',
        'ChainPCC': '2',
        'CLEMS-PCC': '3',
        'LClusterer-PCC': '4',
        'DT-PCC': '5',
        'RF-PCC': '6',

        #MLQ
        'RakelD-PCC': 'RkClust',
        'KMeansClustersPowerSet-PCC': 'kClust',
        'MRQ-Ridge': 'R1',
        'MRQ-MultitaskLasso': 'R2',
        'MRQ-RandomForest': 'R3',
        'MRQ-LinearSVR': 'R4',
        'MRQ-StackedLinearSVR': 'R5',
        'MRQ-ChainedLinearSVR': 'R6'
    }

    # tabular += table.latexTabularT(benchmark_replace=dataset_replace, method_replace=method_replace, side=True)
    tabular += table.latexTabular(benchmark_replace=dataset_replace, method_replace=method_replace)
    tabular += """
        \\bottomrule
        \end{tabularx}%
        }
    """
    with open(path, 'wt') as foo:
        foo.write(tabular)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for multi-label quantification')
    parser.add_argument('--results', type=str, default='./results_generales', metavar='str',
                        help=f'path where to store the results')
    parser.add_argument('--tablepath', type=str, default='./mlqtables', metavar='str',
                        help=f'path where to store the tables')
    parser.add_argument('--subset', type=str, default='all', metavar='str',
                        help='subset of models, default: all, options: [general, mlc, mlq, all]')    
    opt = parser.parse_args()

    assert os.path.exists(opt.results), f'result directory {opt.results} does not exist'
    os.makedirs(opt.tablepath, exist_ok=True)

    qp.environ["SAMPLE_SIZE"] = sample_size
    absolute_error = qp.error.ae
    relative_absolute_error = qp.error.rae

    datasets.remove("jrcall") #FIXME

    #generate_table(f'{opt.tablepath}/npp.ae.tex', protocol='npp', error=absolute_error)
    # generate_table(f'{opt.tablepath}/app.ae.tex', protocol='app', error=absolute_error)
    #generate_table(f'{opt.tablepath}/npp.rae.tex', protocol='npp', error=relative_absolute_error)
    # generate_table(f'{opt.tablepath}/app.rae.tex', protocol='app', error=relative_absolute_error)
    
    if opt.subset == 'all' or opt.subset == 'general':
        models = model_subsets["general"]
        generate_table(f'{opt.tablepath}/cc_nocol.app.ae.tex', protocol='app', error=absolute_error, include=["eCC", "kCC", "-CC", "-StackCC", "-MLCC"], color=False)
        generate_table(f'{opt.tablepath}/cc_col.app.ae.tex', protocol='app', error=absolute_error, include=["eCC", "kCC", "-CC", "-StackCC", "-MLCC"], color=True)
        generate_table(f'{opt.tablepath}/pcc_nocol.app.ae.tex', protocol='app', error=absolute_error, include=["ePCC", "kPCC", "-PCC", "-StackPCC", "-MLPCC"], color=False)
        generate_table(f'{opt.tablepath}/pcc_col.app.ae.tex', protocol='app', error=absolute_error, include=["ePCC", "kPCC", "-PCC", "-StackPCC", "-MLPCC"], color=True)
        generate_table(f'{opt.tablepath}/acc_nocol.app.ae.tex', protocol='app', error=absolute_error, include=["eACC", "kACC", "-ACC", "-StackACC", "-MLACC"], color=False)
        generate_table(f'{opt.tablepath}/acc_col.app.ae.tex', protocol='app', error=absolute_error, include=["eACC", "kACC", "-ACC", "-StackACC", "-MLACC"], color=True)
        generate_table(f'{opt.tablepath}/pacc_nocol.app.ae.tex', protocol='app', error=absolute_error, include=["ePACC", "kPACC", "-PACC", "-StackPACC", "-MLPACC"], color=False)
        generate_table(f'{opt.tablepath}/pacc_col.app.ae.tex', protocol='app', error=absolute_error, include=["ePACC", "kPACC", "-PACC", "-StackPACC", "-MLPACC"], color=True)
    
    if opt.subset == 'all' or opt.subset == 'mlc':
        models = model_subsets["mlc"]
        generate_table(f'{opt.tablepath}/mlc_nocol.app.ae.tex', protocol='app', error=absolute_error, include='all', color=False, prec_mean=3)
        generate_table(f'{opt.tablepath}/mlc_col.app.ae.tex', protocol='app', error=absolute_error, include='all', color=True, prec_mean=3)
    
    if opt.subset == 'all' or opt.subset == 'mlq':
        models = model_subsets["mlq"]
        generate_table(f'{opt.tablepath}/mlq_nocol.app.ae.tex', protocol='app', error=absolute_error, include='all', color=False)
        generate_table(f'{opt.tablepath}/mlq_col.app.ae.tex', protocol='app', error=absolute_error, include='all', color=True)
    
    if opt.subset == 'all' or opt.subset == 'mlq-reg':
        models = model_subsets["mlq-reg"]
        generate_table(f'{opt.tablepath}/mlqreg_nocol.app.ae.tex', protocol='app', error=absolute_error, include='all', color=False, prec_mean=3)
        generate_table(f'{opt.tablepath}/mlqreg_col.app.ae.tex', protocol='app', error=absolute_error, include='all', color=True, prec_mean=3)





    # generate_table(f'{opt.tablepath}/cc0.app.ae.tex', protocol='app', error=absolute_error, include=["eCC", "kCC", "-CC", "-StackCC", "-MLCC"], drift_bin=0)
    # generate_table(f'{opt.tablepath}/cc1.app.ae.tex', protocol='app', error=absolute_error, include=["eCC", "kCC", "-CC", "-StackCC", "-MLCC"], drift_bin=1)
    # generate_table(f'{opt.tablepath}/cc2.app.ae.tex', protocol='app', error=absolute_error, include=["eCC", "kCC", "-CC", "-StackCC", "-MLCC"], drift_bin=2)
    # generate_table(f'{opt.tablepath}/pcc0.app.ae.tex', protocol='app', error=absolute_error, include=["ePCC", "kPCC", "-PCC", "-StackPCC", "-MLPCC"], drift_bin=1)
    # generate_table(f'{opt.tablepath}/pcc1.app.ae.tex', protocol='app', error=absolute_error, include=["ePCC", "kPCC", "-PCC", "-StackPCC", "-MLPCC"], drift_bin=2)
    # generate_table(f'{opt.tablepath}/pcc2.app.ae.tex', protocol='app', error=absolute_error, include=["ePCC", "kPCC", "-PCC", "-StackPCC", "-MLPCC"], drift_bin=0)
    # generate_table(f'{opt.tablepath}/acc0.app.ae.tex', protocol='app', error=absolute_error, include=["eACC", "kACC", "-ACC", "-StackACC", "-MLACC"], drift_bin=0)
    # generate_table(f'{opt.tablepath}/acc1.app.ae.tex', protocol='app', error=absolute_error, include=["eACC", "kACC", "-ACC", "-StackACC", "-MLACC"], drift_bin=1)
    # generate_table(f'{opt.tablepath}/acc2.app.ae.tex', protocol='app', error=absolute_error, include=["eACC", "kACC", "-ACC", "-StackACC", "-MLACC"], drift_bin=2)
    # generate_table(f'{opt.tablepath}/pacc0.app.ae.tex', protocol='app', error=absolute_error, include=["ePACC", "kPACC", "-PACC", "-StackPACC", "-MLPACC"], drift_bin=1)
    # generate_table(f'{opt.tablepath}/pacc1.app.ae.tex', protocol='app', error=absolute_error, include=["ePACC", "kPACC", "-PACC", "-StackPACC", "-MLPACC"], drift_bin=2)
    # generate_table(f'{opt.tablepath}/pacc2.app.ae.tex', protocol='app', error=absolute_error, include=["ePACC", "kPACC", "-PACC", "-StackPACC", "-MLPACC"], drift_bin=0)

    # generate_table(f'{opt.tablepath}/mlq0.app.ae.tex', protocol='app', error=absolute_error, include="all", drift_bin=0)
    # generate_table(f'{opt.tablepath}/mlq1.app.ae.tex', protocol='app', error=absolute_error, include="all", drift_bin=1)
    # generate_table(f'{opt.tablepath}/mlq2.app.ae.tex', protocol='app', error=absolute_error, include="all", drift_bin=2)
    # generate_table(f'{opt.tablepath}/mlq0.app.rae.tex', protocol='app', error=relative_absolute_error, include="all", drift_bin=0)
    # generate_table(f'{opt.tablepath}/mlq1.app.rae.tex', protocol='app', error=relative_absolute_error, include="all", drift_bin=1)
    # generate_table(f'{opt.tablepath}/mlq2.app.rae.tex', protocol='app', error=relative_absolute_error, include="all", drift_bin=2)
    





