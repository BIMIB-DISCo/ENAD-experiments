from maha_extension.model import get_model_transforms
from maha_extension.data import Datasets, TrainValLoader, LabelledTrainLoader, LabelledTestLoader
from maha_extension.base import ALL_NET_TYPES, ALL_DATASETS, ALL_ADV_TYPES, n_classes, n_layers

from net_detector.preprocessing import GroupedScaler
from net_detector.net_detector import NetDetector

from sklearn.utils import shuffle
from sklearn.model_selection import PredefinedSplit
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from skopt import BayesSearchCV
from skopt.space import Real

import os
import logging
import argparse
import pickle
import torch
import numpy as np

def run(
    net_type,
    ds_name,
    adv_type,
    outf='output',
    ocsvm_fname='ocsvm',
    bayes_n_points=1,
    bayes_n_iter=10,
    use_logger=True,
    batch_size = 100,
    pre_computed=True):

    ##################
    # OUTPUT FOLDERS #
    ##################

    # Check if experiment setting folder exists, else create it
    data_outf = f"{outf}/{net_type}_{ds_name}"

    if not os.path.isdir(data_outf):
        os.mkdir(data_outf)

    # Create ocsvm subfolder
    ocsvm_outf = f"{data_outf}/{ocsvm_fname}"
    if not os.path.isdir(ocsvm_outf):
        os.mkdir(ocsvm_outf)

    #########
    # SEEDS #
    #########

    torch.cuda.manual_seed(0)
    torch.cuda.set_device(0)
    np.random.seed(0)

    ##########
    # LOGGER #
    ##########

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fhandler = logging.FileHandler(f'{ocsvm_outf}/scan.log', 'a+', 'utf-8')
    shandler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M")
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)

    logger.addHandler(fhandler)
    logger.addHandler(shandler)

    logger.disables = not use_logger

    #########
    # SETUP #
    #########

    exp_name = f"{net_type}_{ds_name}_{adv_type}"

    logger.info(f"{exp_name}: STARTED")

    # Model
    model, in_transform = get_model_transforms(net_type, ds_name, n_classes(ds_name))

    # Dataset
    ds = Datasets(ds_name, in_transform, net_type, adv_type, data_outf)

    # OCSVM with preprocessing
    clf = Pipeline(steps=[
        ('scaler', GroupedScaler()),
        ('PCA', PCA(whiten=True)),
        ('clf', OneClassSVM(kernel='rbf'))])

    if pre_computed:
        # Param scan not necessary

        layer_det = clf

    else:
        # Define Bayesian hyperparameter optimization procedure

        # Train-validation split
        split = PredefinedSplit([-1]*len(ds.train_loader.dataset) + [1]*len(ds.idxs_val))

        # Search space
        search_spaces = {
            "clf__nu": Real(2**-7, 2**-1, prior='log-uniform', base=2),
            "clf__gamma": Real(2**-15, 2**5, prior='log-uniform', base=2)
        }

        # Bayesian hyperparameter optimization
        layer_det = BayesSearchCV(
            clf,
            search_spaces,
            n_iter=bayes_n_iter,
            n_points=bayes_n_points,
            n_jobs=-1,
            scoring='accuracy',
            cv=split,
            return_train_score=False,
            refit=False,
            verbose=0
        )

    osvm_trainer = NetDetector(n_layers(net_type), layer_det, tqdm=False, logger=logger, exp_name=exp_name, outf=ocsvm_outf, pre_computed=pre_computed)

    ###########
    # FITTING #
    ###########

    osvm_trainer.fit(TrainValLoader(model, ds, batch_size=batch_size), LabelledTrainLoader(model, ds, batch_size=batch_size), ds.adv_test[ds.idxs_train])

    ###########
    # TESTING #
    ###########

    test_scores, output = osvm_trainer.predict(LabelledTestLoader(model, ds, batch_size=batch_size))

    ################
    # SAVE RESULTS #
    ################

    all_output = np.hstack((
        test_scores,
        output,
        np.expand_dims(ds.adv_test[ds.idxs_test], axis=1)))

    with open(f'{ocsvm_outf}/OCSVM_net_detector_{exp_name}.pkl', 'wb') as outp:
        pickle.dump(osvm_trainer, outp, pickle.HIGHEST_PROTOCOL)

    np.save(f"{ocsvm_outf}/OCSVM_{exp_name}", all_output)

    accuracy = accuracy_score(ds.adv_test[ds.idxs_test], output[:,0])
    auroc = roc_auc_score(ds.adv_test[ds.idxs_test], -output[:,1])

    logger.info(f"{exp_name}: ACC = {accuracy}, AUROC = {round(auroc*100, 2)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Net detector trainer.')
    parser.add_argument('--dataset', dest='ds_name', type=str,
        choices=ALL_DATASETS, help='Dataset', required=True)
    parser.add_argument('--net_type', dest='net_type', type=str,
        choices=ALL_NET_TYPES, help='Model', required=True)
    parser.add_argument('--adv_type', dest='adv_type', type=str,
        choices=ALL_ADV_TYPES, help='Attack', required=True)
    parser.add_argument('--n_points', dest='bayes_n_points', type=int, default=1,
        help='Number of parallel evaluation using Skopt BayesSearchCV')
    parser.add_argument('--n_iter', dest='bayes_n_iter', type=int, default=40,
        help='Number of maximum iterations using Skopt BayesSearchCV')
    parser.add_argument('--logger', dest='use_logger', type=bool, default=True,
        help='Use logger')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100,
        help='Use logger')
    parser.add_argument('--outf', dest='outf', type=str, default='output',
        help='Output Folder')
    parser.add_argument('--ocsvm_fname', dest='ocsvm_fname', type=str, default='ocsvm',
        help='Output Folder OCSVM')

    args = parser.parse_args()

    run(**vars(args))

