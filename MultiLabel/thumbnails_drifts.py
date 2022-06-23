from MultiLabel.main_pruebas import get_dataset, SKMULTILEARN_ALL_DATASETS
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from MultiLabel.mldata import MultilabelledCollection
import quapy as qp


def MLAPP_drift(train_prev, test:MultilabelledCollection, sample_size, n_prevalences=21, repeats=10, random_seed=42):
    drifts = []
    with qp.util.temp_seed(random_seed):
        for cat in test.classes_:
            indexes = list(test.artificial_sampling_index_generator(
                sample_size=sample_size, category=cat, n_prevalences=n_prevalences, repeats=repeats, min_df=5)
            )
            if indexes:
                for index in indexes:
                    sample = test.sampling_from_index(index)
                    test_prev = sample.prevalence()
                    drift=qp.error.mae(train_prev, test_prev)
                    drifts.append(drift)
    return drifts


def get_params_for_dataset(dataset_name):
    # DEFAULTS
    n_prevalences = 101
    repeats = 1
    if train.n_classes < 98:
        # DEFAULTS SMALL DATASETS
        n_prevalences = 101
        repeats = 25
        repeats_grid = 5
    elif train.n_classes > 500:
        # DEFAULTS HUGE DATASETS
        n_prevalences = 21

    handcrafted_repeats = {
        "jrcall": 1,
        "delicious": 1,
        "mediamill": 2,
        "birds": 40,
        "genbase": 50,
        "enron": 9,
        "ohsumed": 5,
        "bibtex": 2,
        "reuters21578": 6,
        "tmc2007_500": 5,
        "scene": 18,
        "medical": 15,
        "Corel5k": 6,
        "emotions": 26,
        "yeast": 8,
        "rcv1": 2,
    }

    if dataset_name in handcrafted_repeats.keys():
        repeats = handcrafted_repeats[dataset_name]

    return n_prevalences, repeats

data=[]
for dataset_name in ['reuters21578', 'ohsumed']+SKMULTILEARN_ALL_DATASETS:

    train, test = get_dataset(dataset_name)
    n_prevalences, repeats = get_params_for_dataset(dataset_name)
    drifts = MLAPP_drift(
        train_prev=train.prevalence(),
        test=test,
        sample_size=100,
        n_prevalences=n_prevalences,
        repeats=repeats
    )
    dataset = np.asarray([dataset_name]*len(drifts), dtype=object)
    dataset_data = np.asarray([drifts, dataset]).T
    data.append(dataset_data)

    data.append(dataset_data)

data = np.vstack(data)

df = pd.DataFrame(data, columns=['drift', 'dataset'])

#sns.set_style("whitegrid")
sns.set(font_scale=0.4)
g = sns.FacetGrid(df, col='dataset', col_wrap=5, sharex=False, sharey=False, size=2)
g.map_dataframe(sns.histplot, x="drift", linewidth=0)
g.set_titles(col_template="{col_name}")
g.set(xticks=[])
g.set(yticks=[])
g.tight_layout()
def annotate(data, **kws):
    n = len(data)
    drifts = data['drift'].values
    maxdrift = np.max(drifts)
    mindrift = np.min(drifts)
    p1 = (maxdrift+mindrift) / 3
    p2 = p1*2
    ax = plt.gca()
    ax.axvline(x=p1, color='k', linestyle='dotted', linewidth=1)
    ax.axvline(x=p2, color='k', linestyle='dotted', linewidth=1)
g.map_dataframe(annotate)

plt.show()
g.savefig("drift_thumbnails.pdf")

