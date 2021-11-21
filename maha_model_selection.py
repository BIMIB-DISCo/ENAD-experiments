from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from maha_extension.data import idxs_train_val_test
from maha_extension.base import ALL_NET_TYPES, ALL_DATASETS, ALL_ADV_TYPES, n_layers

import re
import os
import argparse
import numpy as np

def run(ds_name, net_type, adv_type, outf):

    # Change current path
    outf = f"{outf}/{net_type}_{ds_name}"

    for method in ['Mahalanobis', 'LID']:

        print(f'Started model selection for {method}...')

        # LID has int params, Mahalanobis float params
        regex = r'(\d+)' if method == 'LID' else r'(\d+\.\d+)'

        # Extract params from fnames
        params = re.findall(f'{method}_{regex}_{ds_name}_{adv_type}', ''.join(os.listdir(outf)))
        params = np.array(params).astype(int if method == 'LID' else float)

        print(f'{method} params list: {", ".join(np.sort(params.astype(str)))}')

        aurocs = []
        for param in params:
            data = np.load(f'{outf}/{method}_{param}_{ds_name}_{adv_type}.npy')

            # Adv label
            adv =  data[:, -1]
            # Use coherent labelling, i.e. adv -> -1, not adv -> 1
            adv = np.select([adv==0, adv==1], [1, -1])

            # Load spltting idxs
            idxs_train, idxs_val, idxs_test = idxs_train_val_test(len(data))

            #########
            # TRAIN #
            #########

            X_train = data[idxs_train][:,:n_layers(net_type)]
            adv_train = adv[idxs_train]

            lr = LogisticRegressionCV(penalty='l1', solver='liblinear', max_iter=10000, n_jobs=-1)
            lr.fit(X_train, adv_train)

            ########
            # EVAL #
            ########

            X_valid = data[idxs_val][:,:n_layers(net_type)]

            adv_conf_valid = lr.predict_proba(X_valid)[:, 0]

            auroc_valid = roc_auc_score(adv[idxs_val], -adv_conf_valid)

            aurocs.append(auroc_valid)
        aurocs = np.array(aurocs)
        idx_max = np.argmax(aurocs)

        print(f'{method} with param {params[idx_max]} has AUROC = {round(aurocs[idx_max]*100, 2)} on the validation set.')

        X = np.load(f'{outf}/{method}_{params[idx_max]}_{ds_name}_{adv_type}.npy')
        np.save(f'{outf}/{method}_best_{ds_name}_{adv_type}.npy', X)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Net detector trainer.')
    parser.add_argument('--dataset', dest='ds_name', type=str, required=True,
        choices=ALL_DATASETS, help='Dataset')
    parser.add_argument('--net_type', dest='net_type', type=str, required=True,
        choices=ALL_NET_TYPES, help='Model')
    parser.add_argument('--adv_type', dest='adv_type', type=str, required=True,
        choices=ALL_ADV_TYPES, help='Attack')
    parser.add_argument('--outf', dest='outf', type=str, default='output',
        help='Output Folder')

    args = parser.parse_args()

    run(**vars(args))