import os
from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN


class LabeledDataModule(LightningDataModule):
    """DataModule with the logic of labeled dataset used for ENAD. Moreover,
    input data has to be generate through maha_repo:
    https://github.com/pokaxpoka/deep_Mahalanobis_detector.git."""

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        attack_name: str,
        labeled_dir: str,
        datasets_dir: str,
    ):
        """
        Args:
            model_name (str): either densenet or resnet.
            dataset_name (str): either cifar10, cifar100 or svhn.
            attack_name (str): either FGSM, BIM, DeepFool or CWL2.
            labeled_dir (str): location of data generated trough maha_repo.
            datasets_dir (str): location in which torchvision datasets are saved.
        """

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.attack_name = attack_name

        self.datasets_dir = datasets_dir
        self.labeled_dir = labeled_dir

    def setup(self, stage=None):

        # Load train dataset
        self.setup_train_ds()

        # Load labeled dataset
        self.setup_labeled_ds()

    def setup_train_ds(self):
        # 1) Load train dataset from torchvision.datasets.
        # 2) Apply normalization transforms.

        in_transforms = [transforms.ToTensor()]  # type: Any

        if self.model_name == "densenet":
            in_transforms.append(
                transforms.Normalize(
                    (125.3 / 255, 123.0 / 255, 113.9 / 255),
                    (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0),
                )
            )

        elif self.model_name == "resnet":
            in_transforms.append(
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            )
        else:
            raise ValueError()

        in_transforms = transforms.Compose(in_transforms)

        if self.dataset_name == "cifar10":
            self.train_ds = CIFAR10(
                root=self.datasets_dir,
                train=True,
                transform=in_transforms,
                download=False,
            )
        elif self.dataset_name == "svhn":
            self.train_ds = SVHN(
                root=self.datasets_dir,
                split="train",
                transform=in_transforms,
                download=False,
            )
        else:
            raise ValueError()

    def setup_labeled_ds(self):
        # 1) Load pre-generated clean, noisy and adversarial datasets
        # 2) Exclude last batch (for compatibility with original Maha exps)
        # 3) Perform train, val, test split

        labeled_data_dirpath = os.path.join(
            self.labeled_dir, f"{self.model_name}_{self.dataset_name}"
        )
        setting_name = f"{self.model_name}_{self.dataset_name}_{self.attack_name}"

        # Load clean data
        self.labeled_clean_data = torch.load(
            os.path.join(labeled_data_dirpath, f"clean_data_{setting_name}.pth")
        )

        # Exclude last split for compatibility reasons
        self.split_size = 100 * (
            len(self.labeled_clean_data) // 100
        )  # ...to match Maha experiments
        self.labeled_clean_data = self.labeled_clean_data[: self.split_size]

        # Load the adv, noisy data and target labels
        self.labeled_adv_data = torch.load(
            os.path.join(labeled_data_dirpath, f"adv_data_{setting_name}.pth")
        )[: self.split_size]
        self.labeled_noisy_data = torch.load(
            os.path.join(labeled_data_dirpath, f"noisy_data_{setting_name}.pth")
        )[: self.split_size]
        self.labeled_targets = torch.load(
            os.path.join(labeled_data_dirpath, f"label_{setting_name}.pth")
        )[: self.split_size]

        # Create dataset
        labeled_ds = TensorDataset(
            torch.cat(
                (
                    self.labeled_adv_data,
                    self.labeled_clean_data,
                    self.labeled_noisy_data,
                )
            ),
            self.labeled_targets.tile((3,)),
        )

        # Split in train, val, test
        idxs_train, idxs_val, idxs_test = self.train_val_test_split_idxs(
            self.split_size
        )

        self.labeled_train_ds = Subset(labeled_ds, idxs_train)
        self.labeled_val_ds = Subset(labeled_ds, idxs_val)
        self.labeled_test_ds = Subset(labeled_ds, idxs_test)

    def train_dataloader(self, batch_size: int = 100):
        return DataLoader(self.train_ds, batch_size=batch_size)

    def labeled_train_dataloader(self, batch_size: int = 100):
        return DataLoader(self.labeled_train_ds, batch_size=batch_size)

    def labeled_val_dataloader(self, batch_size: int = 100):
        return DataLoader(self.labeled_val_ds, batch_size=batch_size)

    def labeled_test_dataloader(self, batch_size: int = 100):
        return DataLoader(self.labeled_test_ds, batch_size=batch_size)

    @staticmethod
    def train_val_test_split_idxs(split_size: int, trainval_frac: float = 0.1):

        train_val_span = int(split_size * trainval_frac)
        idxs_trainval = np.concatenate(
            [
                np.arange(train_val_span),
                np.arange(split_size, split_size + train_val_span),
                np.arange(2 * split_size, 2 * split_size + train_val_span),
            ]
        )

        idxs_test = np.delete(np.arange(split_size * 3), idxs_trainval)

        pivot = int(len(idxs_trainval) / 6)
        idxs_train = np.concatenate(
            [
                idxs_trainval[:pivot],
                idxs_trainval[2 * pivot : 3 * pivot],
                idxs_trainval[4 * pivot : 5 * pivot],
            ]
        )
        idxs_val = np.concatenate(
            [
                idxs_trainval[pivot : 2 * pivot],
                idxs_trainval[3 * pivot : 4 * pivot],
                idxs_trainval[5 * pivot :],
            ]
        )

        return idxs_train, idxs_val, idxs_test
