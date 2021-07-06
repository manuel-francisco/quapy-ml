from typing import Union, Callable

import numpy as np
import quapy as qp
from MultiLabel.mlquantification import MLAggregativeQuantifier
from mldata import MultilabelledCollection



def ml_natural_prevalence_evaluation(model,
                                     test:MultilabelledCollection,
                                     sample_size,
                                     repeats=100,
                                     error_metric:Union[str,Callable]='mae',
                                     random_seed=42):

    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)

    assert hasattr(error_metric, '__call__'), 'invalid error function'

    test_batch_fn = _test_quantification_batch
    if isinstance(model, MLAggregativeQuantifier):
        test = MultilabelledCollection(model.preclassify(test.instances), test.labels)
        test_batch_fn = _test_aggregation_batch

    with qp.util.temp_seed(random_seed):
        test_indexes = list(test.natural_sampling_index_generator(sample_size=sample_size, repeats=repeats))

    errs = test_batch_fn(tuple([model, test, test_indexes, error_metric]))
    return np.mean(errs)


def ml_artificial_prevalence_evaluation(model,
                                        test:MultilabelledCollection,
                                        sample_size,
                                        n_prevalences=21,
                                        repeats=10,
                                        error_metric:Union[str,Callable]='mae',
                                        random_seed=42):

    if isinstance(error_metric, str):
        error_metric = qp.error.from_name(error_metric)

    assert hasattr(error_metric, '__call__'), 'invalid error function'

    test_batch_fn = _test_quantification_batch
    if isinstance(model, MLAggregativeQuantifier):
        test = MultilabelledCollection(model.preclassify(test.instances), test.labels)
        test_batch_fn = _test_aggregation_batch

    test_indexes = []
    with qp.util.temp_seed(random_seed):
        for cat in test.classes_:
            test_indexes.append(list(test.artificial_sampling_index_generator(sample_size=sample_size,
                                                                              category=cat,
                                                                              n_prevalences=n_prevalences,
                                                                              repeats=repeats)))

    args = [(model, test, indexes, error_metric) for indexes in test_indexes]
    macro_errs = qp.util.parallel(test_batch_fn, args, n_jobs=-1)

    return np.mean(macro_errs)


def _test_quantification_batch(args):
    model, test, indexes, error_metric = args
    errs = []
    for index in indexes:
        sample = test.sampling_from_index(index)
        estim_prevs = model.quantify(sample.instances)
        true_prevs = sample.prevalence()
        errs.append(error_metric(true_prevs, estim_prevs))
    return errs


def _test_aggregation_batch(args):
    model, preclassified_test, indexes, error_metric = args
    errs = []
    for index in indexes:
        sample = preclassified_test.sampling_from_index(index)
        estim_prevs = model.aggregate(sample.instances)
        true_prevs = sample.prevalence()
        errs.append(error_metric(true_prevs, estim_prevs))
    return errs