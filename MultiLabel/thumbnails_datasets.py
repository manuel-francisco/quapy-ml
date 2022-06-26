from MultiLabel.main_pruebas import get_dataset, SKMULTILEARN_ALL_DATASETS
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data=[]
for dataset_name in ['reuters21578', 'ohsumed']+SKMULTILEARN_ALL_DATASETS:

    train, _ = get_dataset(dataset_name)

    n_entries = train.n_classes
    prevs = np.sort(np.asarray(train.labels.mean(axis=0)))[::-1]
    labels = np.arange(n_entries)
    dataset = np.asarray([dataset_name]*n_entries, dtype=object)
    dataset_data = np.asarray([labels, prevs, dataset]).T

    data.append(dataset_data)

data = np.vstack(data)

df = pd.DataFrame(data, columns=['cats', 'prevs', 'dataset'])

#sns.set_style("whitegrid")
sns.set(font_scale=0.4)
g = sns.FacetGrid(df, col='dataset', col_wrap=5, sharex=False, sharey=False, size=2)
g.map_dataframe(sns.barplot, x="cats", y='prevs', linewidth=0)
g.set_titles(col_template="{col_name}")
g.set(xticks=[])
g.set(yticks=[])
g.tight_layout()

plt.show()
g.savefig("drift_prevalences.pdf")

