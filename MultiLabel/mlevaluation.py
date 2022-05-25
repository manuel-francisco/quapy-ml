from typing import Union, Callable

import numpy as np
import quapy as qp
from MultiLabel.mlquantification import MLAggregativeQuantifier
from mldata import MultilabelledCollection
import itertools
from tqdm import tqdm


def check_error_str(error_metric):
    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)

    assert hasattr(error_metric, '__call__'), 'invalid error function'
    return error_metric


def _ml_prevalence_predictions(model,
                               test: MultilabelledCollection,
                               test_indexes):

    predict_batch_fn = _predict_quantification_batch
    if isinstance(model, MLAggregativeQuantifier):
        test = MultilabelledCollection(model.preclassify(test.instances), test.labels)
        predict_batch_fn = _predict_aggregative_batch

    args = tuple([model, test, test_indexes])
    true_prevs, estim_prevs = predict_batch_fn(args)
    return true_prevs, estim_prevs


def ml_natural_prevalence_prediction(model,
                                     test:MultilabelledCollection,
                                     sample_size,
                                     repeats=100,
                                     random_seed=42):

    with qp.util.temp_seed(random_seed):
        test_indexes = list(test.natural_sampling_index_generator(sample_size=sample_size, repeats=repeats))

    return _ml_prevalence_predictions(model, test, test_indexes)


def ml_natural_prevalence_evaluation(model,
                                     test:MultilabelledCollection,
                                     sample_size,
                                     repeats=100,
                                     error_metric:Union[str,Callable]='mae',
                                     random_seed=42):

    error_metric = check_error_str(error_metric)

    true_prevs, estim_prevs = ml_natural_prevalence_prediction(model, test, sample_size, repeats, random_seed)

    errs = [error_metric(true_prev_i, estim_prev_i) for true_prev_i, estim_prev_i in zip(true_prevs, estim_prevs)]
    return np.mean(errs)


def ml_artificial_prevalence_prediction(model,
                                        test:MultilabelledCollection,
                                        sample_size,
                                        n_prevalences=21,
                                        repeats=10,
                                        random_seed=42,
                                        n_jobs=-1):

    nested_test_indexes = []
    with qp.util.temp_seed(random_seed):
        for cat in test.classes_:
            indexes = list(test.artificial_sampling_index_generator(
                sample_size=sample_size, category=cat, n_prevalences=n_prevalences, repeats=repeats, min_df=5)
            )
            if indexes:
                nested_test_indexes.append(indexes)
    def _predict_batch(test_indexes):
        return _ml_prevalence_predictions(model, test, test_indexes)

    predictions = qp.util.parallel(_predict_batch, nested_test_indexes, n_jobs=n_jobs)
    trues, estims = zip(*predictions)
    true_prevs = list(itertools.chain.from_iterable(trues))
    estim_prevs = list(itertools.chain.from_iterable(estims))
    return true_prevs, estim_prevs


def ml_artificial_prevalence_evaluation(model,
                                        test:MultilabelledCollection,
                                        sample_size,
                                        n_prevalences=21,
                                        repeats=10,
                                        error_metric:Union[str,Callable]='mae',
                                        random_seed=42):

    error_metric = check_error_str(error_metric)

    true_prevs, estim_prevs = ml_artificial_prevalence_prediction(model, test, sample_size, n_prevalences, repeats, random_seed)

    errs = [error_metric(true_prev_i, estim_prev_i) for true_prev_i, estim_prev_i in zip(true_prevs, estim_prevs)]
    return np.mean(errs)


def _predict_quantification_batch(args):
    model, test, indexes = args
    return __predict_batch_fn(args, model.quantify)


def _predict_aggregative_batch(args):
    model, test, indexes = args
    return __predict_batch_fn(args, model.aggregate)


def __predict_batch_fn(args, quant_fn):
    model, test, indexes = args
    trues, estims = [], []
    for index in indexes:
        sample = test.sampling_from_index(index)
        estims.append(quant_fn(sample.instances))
        trues.append(sample.prevalence())
    return trues, estims

