import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Normalize


class AttrDataModule(LightningDataModule):
    def __init__(
        self,
        mode: str,
        model_name: str,
        dataset_name: str,
        adv_name: str,
        attr_name: str,
        target: int,
        data_dir: str,
        train_batch_size: int = 128,
        num_workers: int = 4,
    ):
        """
        Datamodule of feature attributions for ExAD.

        Args:
            mode (str): CNN or AE.
            model_name (str): densenet or resnet.
            dataset_name (str): cifar10 or svhn.
            adv_name (str): either FGSM, BIM, DeepFool or CWL2
            attr_name (str): attribution name, generate throw attr_gen.py
            target (int): target class, for CIFAR10 and SVHN between 0 and 9.
            data_dir (str): location of data generated through attr_gen.py
            train_batch_size (int, optional): Defaults to 128.
            num_workers (int, optional): Defaults to 4.
        """

        super().__init__()
        self.mode = mode
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.adv_name = adv_name
        self.attr_name = attr_name
        self.target = target
        self.data_dir = data_dir

        self.train_batch_size = train_batch_size
        self.num_workers = num_workers

        self.save_hyperparameters()

    def prepare_data(self):
        fname = (
            f"{self.model_name}_{self.dataset_name}_{self.adv_name}_{self.attr_name}"
        )

        if self.mode == "AE":
            train_fname = (
                f"{self.model_name}_{self.dataset_name}_{self.attr_name}_train"
            )
            train_data = (
                torch.load(os.path.join(self.data_dir, f"{train_fname}.pth"))
                .float()
                .detach()
            )
            self.train_targets = torch.load(
                os.path.join(self.data_dir, f"{train_fname}_targets.pth")
            )
        else:
            train_data = (
                torch.load(os.path.join(self.data_dir, f"{fname}_ltrain.pth"))
                .float()
                .detach()
            )
            self.train_targets = torch.load(
                os.path.join(self.data_dir, f"{fname}_ltrain_targets.pth")
            )

        normalize = Normalize(
            train_data.mean(dim=[0, 2, 3]), train_data.std(dim=[0, 2, 3])
        )
        self.train_data = normalize(train_data)

        val_data = (
            torch.load(os.path.join(self.data_dir, f"{fname}_lval.pth"))
            .float()
            .detach()
        )
        val_data.requires_grad = False
        self.val_data = normalize(val_data)

        test_data = (
            torch.load(os.path.join(self.data_dir, f"{fname}_ltest.pth"))
            .float()
            .detach()
        )
        test_data.requires_grad = False
        self.test_data = normalize(test_data)

        self.val_targets = torch.load(
            os.path.join(self.data_dir, f"{fname}_lval_targets.pth")
        )
        self.test_targets = torch.load(
            os.path.join(self.data_dir, f"{fname}_ltest_targets.pth")
        )

    def setup(self, stage=None):

        for stage_ds_name in ["train", "val", "test"]:
            attributions = getattr(self, f"{stage_ds_name}_data")
            targets = getattr(self, f"{stage_ds_name}_targets")

            setattr(
                self,
                f"{stage_ds_name}_ds",
                self.gen_dataset(attributions, targets, stage_ds_name),
            )

    def gen_dataset(self, attributions, targets, stage):

        mask = targets == self.target

        if stage == "train" and self.mode == "AE":
            adv_labels = [0] * len(attributions)
        else:
            adv_split = len(attributions) // 3
            adv_labels = [1] * adv_split + [0] * (adv_split * 2)

        attributions = attributions[mask]
        adv_labels = torch.tensor(adv_labels)[mask]

        print(stage, len(attributions))

        return TensorDataset(attributions, adv_labels)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,  # type: ignore
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=100)  # type: ignore

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=100)  # type: ignore
