from data.dataset import Dataset
from tqdm import tqdm
import os
import numpy as np


def write_data(documents, labels, fout):
    print(f'there are {len(documents)} documents')
    written, empty = 0, 0
    with open(fout, 'wt') as foo:
        for doc, label in tqdm(list(zip(documents, labels))):
            doc = doc.replace('\t', ' ').replace('\n', ' ').strip()
            label = np.squeeze(np.asarray(label.todense()))
            label = ' '.join([f'{x}' for x in label])
            if doc:
                foo.write(f'{label}\t{doc}\n')
                written += 1
            else:
                foo.write(f'{label}\tempty document\n')
                empty += 1
    print(f'written = {written}')
    print(f'empty = {empty}')


for dataset_name in ['reuters21578', 'ohsumed', 'jrcall', 'rcv1', 'wipo-sl-sc']: #'20newsgroups'

    dataset = Dataset.load(dataset_name=dataset_name, pickle_path=f'../pickles/{dataset_name}.pickle').show()

    os.makedirs(f'../leam/{dataset_name}', exist_ok=True)
    write_data(dataset.devel_raw, dataset.devel_labelmatrix, f'../leam/{dataset_name}/train.csv')
    #write_data(dataset.test_raw, dataset.test_labelmatrix, f'../leam/{dataset_name}/test.csv')
    print('done')

