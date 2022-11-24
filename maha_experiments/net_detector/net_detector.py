import numpy as np
import pickle
import json

from tqdm.auto import tqdm

from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import clone

from skopt.callbacks import DeltaYStopper


class NetDetector:
    def __init__(
        self,
        n_layers,
        detector_class,
        tqdm=True,
        logger=None,
        exp_name="",
        outf="",
        pre_computed=True,
        pre_computed_path="../pre_computed/OCSVM_best_params.json",
    ):

        self.detector_class = detector_class

        self.tqdm = not tqdm

        self.n_layers = n_layers

        self.logger = logger

        self.exp_name = exp_name

        self.outf = outf

        self.pre_computed = pre_computed

        self.pre_computed_path = pre_computed_path

    def fit(self, dl_train, dl_unseen_train, adv_unseen_train):

        self.logger.info(f"{self.exp_name}: training layer detectors...")

        # Train one-class layer detectors
        self.detectors = self.train_layer_detectors(dl_train)

        # Extract unseen_train layer scores
        train_scores = self.get_layer_scores(dl_unseen_train)

        self.logger.info(f"{self.exp_name}: training final logistic...")

        # Train logistic regression
        self.lr = self.train_logistic_regression(train_scores, adv_unseen_train)

    def predict(self, data_loader):

        # Extract layer scores
        predicted_scores = self.get_layer_scores(data_loader)

        metrics = self.get_final_metrics(predicted_scores)

        return predicted_scores, metrics

    def train_layer_detectors(self, data_loader):

        detectors = []
        for layer_idx in tqdm(
            range(self.n_layers),
            disable=not self.tqdm,
            desc="Training layer detectors...",
        ):

            self.logger.info(f"{self.exp_name}: started layer {layer_idx}")

            X_train, X_valid, y_train, y_valid, adv_train, adv_valid = data_loader(
                layer_idx
            )

            X = np.concatenate((X_train, X_valid))
            y = np.concatenate((y_train, y_valid))
            adv = np.concatenate((adv_train, adv_valid))

            if self.pre_computed:
                # Use pre-computed hyperparameters

                with open(self.pre_computed_path, "r") as fname:
                    params = json.load(fname)

                detector = clone(self.detector_class)

                best_params = params[f"{self.exp_name}_{layer_idx}"]
                detector = detector.set_params(**best_params).fit(
                    np.c_[X_train, y_train]
                )

            else:
                # Search best hyperparameters, retrain on trainset only

                srch = clone(self.detector_class)
                srch.fit(np.c_[X, y], adv, callback=DeltaYStopper(0.02, n_best=10))

                with open(
                    f"{self.outf}/bayes_{self.exp_name}_{layer_idx}.pkl", "wb"
                ) as outp:
                    pickle.dump(srch, outp, pickle.HIGHEST_PROTOCOL)

                detector = srch.estimator.set_params(**srch.best_params_).fit(
                    np.c_[X_train, y_train]
                )

                self.logger.info(
                    f"{self.exp_name}: ACC = {srch.best_score_}, BEST_PARAMS = {srch.best_params_}"
                )

            detectors.append(detector)

        return detectors

    def get_layer_scores(self, data_loader):

        scores = []
        for layer_idx in tqdm(
            range(self.n_layers),
            disable=not self.tqdm,
            desc="Extracting layer scores...",
            leave=False,
        ):

            X, y = data_loader(layer_idx)

            scores.append(self.detectors[layer_idx].decision_function(np.c_[X, y]))

        return np.vstack(scores).T

    def train_logistic_regression(self, X, adv):

        lr = LogisticRegressionCV(
            penalty="l1", solver="liblinear", max_iter=10000, n_jobs=-1
        )
        lr.fit(X, adv)

        return lr

    def get_final_metrics(self, X):

        preds = self.lr.predict(X)
        probas = self.lr.predict_proba(X)

        return np.array([preds, probas[:, 0]]).T
