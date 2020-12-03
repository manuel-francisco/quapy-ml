from sklearn.metrics import f1_score
from settings import SAMPLE_SIZE


def f1e(y_true, y_pred):
    return 1. - f1_score(y_true, y_pred, average='macro')


def acce(y_true, y_pred):
    acc = (y_true == y_pred).mean()
    return 1. - acc


def mae(prevs, prevs_hat):
    return ae(prevs, prevs_hat).mean()


def ae(p, p_hat):
    assert p.shape == p_hat.shape, 'wrong shape'
    return abs(p_hat-p).mean(axis=-1)


def mrae(p, p_hat, eps=1./(2. * SAMPLE_SIZE)):
    return rae(p, p_hat, eps).mean()


def rae(p, p_hat, eps=1./(2. * SAMPLE_SIZE)):
    p = smooth(p, eps)
    p_hat = smooth(p_hat, eps)
    return (abs(p-p_hat)/p).mean(axis=-1)


def smooth(p, eps):
    n_classes = p.shape[-1]
    return (p+eps)/(eps*n_classes + 1)


CLASSIFICATION_ERROR = {f1e, acce}
QUANTIFICATION_ERROR = {mae, mrae}

f1_error = f1e
acc_error = acce
mean_absolute_error = mae
absolute_error = ae
mean_relative_absolute_error = mrae
relative_absolute_error = rae

