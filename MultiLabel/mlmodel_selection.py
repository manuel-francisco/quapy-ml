import itertools
import signal
from copy import deepcopy
from typing import Union, Callable

import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.evaluation import artificial_prevalence_prediction, natural_prevalence_prediction
from quapy.method.aggregative import BaseQuantifier
from MultiLabel.mlquantification import MLQuantifier
from MultiLabel.mldata import MultilabelledCollection
from MultiLabel.mlevaluation import ml_artificial_prevalence_prediction, ml_natural_prevalence_prediction

import numpy as np


# FIXME: copypasted from quapy.model_selection with adhoc changes
class MLGridSearchQ(MLQuantifier):
    def __init__(self,
                 model: MLQuantifier,
                 param_grid: dict,
                 sample_size: int,
                 protocol='app',
                 n_prevalences: int = 21,
                 repeats: int = 10,
                 error: Union[Callable, str] = qp.error.mae,
                 refit=True,
                 val_split=0.4,
                 n_jobs=1,
                 random_seed=42,
                 timeout=-1,
                 verbose=False):

        self.model = model
        self.param_grid = param_grid
        self.sample_size = sample_size
        self.protocol = protocol.lower()
        self.n_prevalences = n_prevalences
        self.repeats = repeats
        self.refit = refit
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.timeout = timeout
        self.verbose = verbose
        self.__check_error(error)
        assert self.protocol in {'app', 'npp'}, \
            'unknown protocol; valid ones are "app" or "npp" for the "artificial" or the "natural" prevalence protocols'
        if self.protocol == 'npp':
            if self.n_repetitions is None or self.n_repetitions == 1:
                if self.eval_budget is not None:
                    print(f'[warning] when protocol=="npp" the parameter n_repetitions should be indicated '
                          f'(and not eval_budget). Setting n_repetitions={self.eval_budget}...')
                    self.n_repetitions = self.eval_budget
                else:
                    raise ValueError(f'when protocol=="npp" the parameter n_repetitions should be indicated '
                                     f'(and should be >1).')
            if self.n_prevpoints is not None:
                print('[warning] n_prevpoints has been set along with the npp protocol, and will be ignored')

    def sout(self, msg):
        if self.verbose:
            print(f'[{self.__class__.__name__}]: {msg}')

    def __check_training_validation(self, training, validation):
        if isinstance(validation, MultilabelledCollection):
            return training, validation
        elif isinstance(validation, float):
            assert 0. < validation < 1., 'validation proportion should be in (0,1)'
            training, validation = training.train_test_split(train_prop=1 - validation)
            return training, validation
        else:
            raise ValueError(f'"validation" must either be a MultilabelledCollection or a float in (0,1) indicating the'
                             f'proportion of training documents to extract (type found: {type(validation)})')

    def __check_error(self, error):
        if error in qp.error.QUANTIFICATION_ERROR:
            self.error = error
        elif isinstance(error, str):
            self.error = qp.error.from_name(error)
        elif hasattr(error, '__call__'):
            self.error = error
        else:
            raise ValueError(f'unexpected error type; must either be a callable function or a str representing\n'
                             f'the name of an error function in {qp.error.QUANTIFICATION_ERROR_NAMES}')

    def __generate_predictions(self, model, val_split):
        commons = {
            'n_repetitions': self.repeats,
            'n_jobs': self.n_jobs,
            'random_seed': self.random_seed,
            'verbose': False
        }
        if self.protocol == 'app' and isinstance(val_split, MultilabelledCollection):
            return ml_artificial_prevalence_prediction(
                model, val_split, self.sample_size,
                n_prevalences=self.n_prevalences,
                repeats=self.repeats,
                n_jobs=self.n_jobs,
                # **commons
            )
        elif self.protocol == 'app' and isinstance(val_split, LabelledCollection):
            return artificial_prevalence_prediction(
                model, val_split, self.sample_size,
                n_prevpoints=self.n_prevalences,
                #**commons
            )
        elif self.protocol == 'npp':
            return ml_natural_prevalence_prediction(
                model, val_split, self.sample_size,
                **commons)
        else:
            raise ValueError('unknown protocol')

    def fit(self, training: MultilabelledCollection, val_split: Union[MultilabelledCollection, float] = None):
        """
        :param training: the training set on which to optimize the hyperparameters
        :param val_split: either a LabelledCollection on which to test the performance of the different settings, or
        a float in [0,1] indicating the proportion of labelled data to extract from the training set
        """
        if val_split is None:
            val_split = self.val_split
        training, val_split = self.__check_training_validation(training, val_split)
        assert isinstance(self.sample_size, int) and self.sample_size > 0, 'sample_size must be a positive integer'

        params_keys = list(self.param_grid.keys())
        params_values = list(self.param_grid.values())

        model = self.model
        n_jobs = self.n_jobs

        if self.timeout > 0:
            def handler(signum, frame):
                self.sout('timeout reached')
                raise TimeoutError()

            signal.signal(signal.SIGALRM, handler)

        self.sout(f'starting optimization with n_jobs={n_jobs}')
        self.param_scores_ = {}
        self.best_score_ = None
        some_timeouts = False
        for values in itertools.product(*params_values):
            params = dict({k: values[i] for i, k in enumerate(params_keys)})

            if self.timeout > 0:
                signal.alarm(self.timeout)

            try:
                # overrides default parameters with the parameters being explored at this iteration
                model.set_params(**params)
                model.fit(training)
                true_prevalences, estim_prevalences = self.__generate_predictions(model, val_split)
                errs = [self.error(true_prev_i, estim_prev_i) for true_prev_i, estim_prev_i in zip(true_prevalences, estim_prevalences)]
                score = np.mean(errs)
                self.sout(f'checking hyperparams={params} got {self.error.__name__} score {score:.5f}')
                if self.best_score_ is None or score < self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params
                    self.best_model_ = deepcopy(model)
                self.param_scores_[str(params)] = score

                if self.timeout > 0:
                    signal.alarm(0)
            except TimeoutError:
                print(f'timeout reached for config {params}')
                some_timeouts = True
            # except Exception as e:
            #     print(f"it seems like there's been a problem with this set of params {params}. Skiping")
            #     print(e)
            #     continue

        if self.best_score_ is None and some_timeouts:
            raise TimeoutError('all jobs took more than the timeout time to end')

        self.sout(f'optimization finished: best params {self.best_params_} (score={self.best_score_:.5f})')
        # model.set_params(**self.best_params_)
        # self.best_model_ = deepcopy(model)

        if self.refit:
            self.sout(f'refitting on the whole development set')
            self.best_model_.fit(training + val_split)

        return self

    def quantify(self, instances):
        assert hasattr(self, 'best_model_'), 'quantify called before fit'
        return self.best_model_.quantify(instances)
    
    def preclassify(self, instances):
        assert hasattr(self, 'best_model_'), 'quantify called before fit'
        return self.best_model_.preclassify(instances)
    
    def aggregate(self, instances):
        assert hasattr(self, 'best_model_'), 'quantify called before fit'
        return self.best_model_.aggregate(instances)

    @property
    def classes_(self):
        return self.best_model_.classes_

    def set_params(self, **parameters):
        self.param_grid = parameters

    def get_params(self, deep=True):
        if hasattr(self, 'best_model_'):
            return self.best_model_.get_params()
        raise ValueError('get_params called before fit')

    def best_model(self):
        if hasattr(self, 'best_model_'):
            return self.best_model_
        raise ValueError('best_model called before fit')
