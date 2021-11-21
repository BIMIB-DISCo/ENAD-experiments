from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score

from maha_extension.base import ALL_NET_TYPES, ALL_DATASETS, ALL_ADV_TYPES, n_classes, n_layers
from maha_extension.model import get_model_transforms
from maha_extension.data import Datasets, LabelledTrainLoader

from utils import aupr

import numpy as np
import pickle as pkl
import argparse

def run(
    ds_name,
    net_type,
    adv_type,
    outf='output',
    ocsvm_fname='ocsvm'):

    # Change current path
    outf = f"{outf}/{net_type}_{ds_name}"

    lid_data = np.load(f"{outf}/LID_best_{ds_name}_{adv_type}.npy")
    maha_data = np.load(f"{outf}/Mahalanobis_best_{ds_name}_{adv_type}.npy")

    model, in_transform = get_model_transforms(net_type, ds_name, n_classes(ds_name))
    dataset = Datasets(ds_name, in_transform, net_type, adv_type, outf)
    data_loader = LabelledTrainLoader(model, dataset)

    with open(f'{outf}/{ocsvm_fname}/OCSVM_net_detector_{net_type}_{ds_name}_{adv_type}.pkl', 'rb') as fname:
        ocsvm_det = pkl.load(fname)

    ocsvm_train_data = ocsvm_det.get_layer_scores(data_loader)

    X_train = np.c_[
        lid_data[dataset.idxs_train][:, :n_layers(net_type)],
        maha_data[dataset.idxs_train][:, :n_layers(net_type)],
        ocsvm_train_data[:, :n_layers(net_type)]]

    #########
    # Train #
    #########

    lr = LogisticRegressionCV(
        penalty='l1', solver='liblinear', max_iter=10000, n_jobs=-1)
    lr.fit(X_train, dataset.adv_test[dataset.idxs_train])

    ########
    # Test #
    ########

    ocsv_test = np.load(f"{outf}/{ocsvm_fname}/OCSVM_{net_type}_{ds_name}_{adv_type}.npy")

    X_test = np.c_[
        lid_data[dataset.idxs_test][:, :n_layers(net_type)],
        maha_data[dataset.idxs_test][:, :n_layers(net_type)],
        ocsv_test[:, :n_layers(net_type)]]

    adv_conf_test = lr.predict_proba(X_test)[:, 0]

    auroc_test = roc_auc_score(dataset.adv_test[dataset.idxs_test], -adv_conf_test)
    aupr_test = aupr(dataset.adv_test[dataset.idxs_test], adv_conf_test, pos_label=-1)

    print(f"EAD test performances are: AUROC = {round(auroc_test*100, 2)} and AUPR = {round(aupr_test*100, 2)}.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Net detector trainer.')
    parser.add_argument('--dataset', dest='ds_name', type=str, required=True,
        choices=ALL_DATASETS, help='Dataset')
    parser.add_argument('--net_type', dest='net_type', type=str, required=True,
        choices=ALL_NET_TYPES, help='Model')
    parser.add_argument('--adv_type', dest='adv_type', type=str, required=True,
        choices=ALL_ADV_TYPES, help='Attack')
    parser.add_argument('--outf', dest='outf', type=str, default='output',
        help='Output Folder.')
    parser.add_argument('--ocsvm_fname', dest='ocsvm_fname', type=str, default='ocsvm',
        help='Output Folder OCSVM.')

    args = parser.parse_args()

    run(**vars(args))