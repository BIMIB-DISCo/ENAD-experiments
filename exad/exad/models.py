import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import (
    Conv2d,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
    Unflatten,
)
from torchmetrics import Accuracy, AUROC, CatMetric
from torchmetrics.utilities.data import dim_zero_cat
from typing import List, Union, Any
import os


class ExAD_CNN(Sequential):
    def __init__(self):
        super().__init__()

        self.add_module("conv0", Conv2d(3, 32, 3, padding="same"))
        self.add_module("relu0", ReLU(inplace=True))
        self.add_module("conv1", Conv2d(32, 64, 3, padding="same"))
        self.add_module("relu1", ReLU(inplace=True))
        self.add_module("maxpool0", MaxPool2d(2, 2))

        self.add_module("conv2", Conv2d(64, 128, 3, padding="same"))
        self.add_module("relu2", ReLU(inplace=True))
        self.add_module("conv3", Conv2d(128, 128, 3, padding="same"))
        self.add_module("relu3", ReLU(inplace=True))
        self.add_module("maxpool1", MaxPool2d(2, 2))

        self.add_module("flatten", Flatten(start_dim=1))

        self.add_module("dense0", Linear(128 * 8 * 8, 512))
        self.add_module("dense1", Linear(512, 64))
        self.add_module("out", Linear(64, 2))


class ExAD_AE(Sequential):
    def __init__(
        self,
        image_size: int = 32,
        hidden_sizes: List[int] = [1024, 100, 1024],
    ):
        super().__init__()

        self.add_module("flatten", Flatten(start_dim=1))

        input_size = 3 * image_size**2
        hidden_sizes = [input_size] + hidden_sizes
        for idx in range(len(hidden_sizes) - 1):
            self.add_module(
                f"dense{idx}", Linear(hidden_sizes[idx], hidden_sizes[idx + 1])
            )
            self.add_module(f"relu{idx}", ReLU(inplace=True))

        self.add_module(f"out", Linear(hidden_sizes[-1], input_size))

        self.add_module(
            "unflatten", Unflatten(dim=1, unflattened_size=(3, image_size, image_size))
        )


class ExAD_Module(pl.LightningModule):

    model: torch.nn.Module

    def __init__(self, mode: str, lr: float, setting_name: str, outdir: str):
        super().__init__()

        self.mode = mode
        self.lr = lr
        self.setting_name = setting_name
        self.outdir = outdir

        for stage in ["train", "val", "test"]:
            self.__setattr__(f"{stage}_auroc", AUROC())

            if not (self.mode == "AE" and stage != "test"):
                # There is only test_acc for AE
                self.__setattr__(f"{stage}_acc", Accuracy(num_classes=2))

        self.test_pred = CatMetric()
        self.test_ytrue = CatMetric()

        if self.mode == "AE":
            # Find threshold in validation set
            self.val_fpr_thre = Quantile(0.95)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.stage_step(batch, batch_idx, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.stage_step(batch, batch_idx, "val")

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.stage_step(batch, batch_idx, "test")

        return loss

    def stage_step(self, batch, batch_idx, stage):
        if self.mode == "CNN":
            loss = self.stage_step_cnn(batch, batch_idx, stage)
        else:
            loss = self.stage_step_ae(batch, batch_idx, stage)

        self.log(
            f"{stage}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=stage == "train",
        )

        return loss

    def stage_step_cnn(self, batch, batch_idx, stage):
        x, y = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        acc_metric = self.__getattr__(f"{stage}_acc")
        acc_metric.update(y_hat, y)  # type: ignore
        self.log(
            f"{stage}_acc",
            acc_metric,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=stage == "val",
        )

        auroc_metric = self.__getattr__(f"{stage}_auroc")
        auroc_metric.update(F.softmax(y_hat, 1)[:, 1], y)  # type: ignore
        self.log(
            f"{stage}_auroc",
            auroc_metric,  # type: ignore
            on_step=False,
            on_epoch=True,
            prog_bar=stage == "val",
        )

        if stage == "test":
            self.test_pred.update(y_hat.argmax(1))
            self.test_ytrue.update(y)

        return loss

    def stage_step_ae(self, batch, batch_idx, stage):
        x, y = batch
        x_hat = self(x)

        loss = F.mse_loss(x_hat, x)

        sample_loss = F.mse_loss(x_hat, x, reduction="none")
        sample_loss = sample_loss.mean([1, 2, 3])

        if stage != "train":
            auroc_metric = self.__getattr__(f"{stage}_auroc")
            auroc_metric.update(sample_loss, y)  # type: ignore
            self.log(
                f"{stage}_auroc",
                auroc_metric,  # type: ignore
                on_step=False,
                on_epoch=True,
                prog_bar=stage == "val",
            )

        if stage == "val":
            # Update values to comute threshold

            self.val_fpr_thre.update(sample_loss)

        if stage == "test":
            # Update and log accuracy

            # 1 for adv, 0 for normal
            y_hat = (sample_loss > self.curr_threshold).int()

            self.test_pred.update(y_hat)
            self.test_ytrue.update(y)

            acc_metric = self.__getattr__(f"{stage}_acc")
            acc_metric.update(y_hat, y)  # type: ignore
            self.log(
                f"{stage}_acc", acc_metric, on_step=False, on_epoch=True  # type: ignore
            )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_validation_epoch_end(self):
        if self.mode == "AE":
            self.on_validation_epoch_end_ae()

    def on_validation_epoch_end_ae(self):
        # On validation end, compute and save threshold

        curr_threshold = self.val_fpr_thre.compute()
        self.val_fpr_thre.reset()
        self.log(
            f"val_fpr_threshold",
            curr_threshold,  # type: ignore
        )
        self.curr_threshold = curr_threshold

    def on_test_epoch_end(self):
        # On test end, save accumulated predictions to then ensemble

        outdir = os.path.join(
            self.outdir,
            f"{'_'.join(self.setting_name.split('_')[:3])}_{self.mode}",
        )

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        predictions = self.test_pred.compute()
        targets = self.test_ytrue.compute()

        torch.save(
            predictions,
            os.path.join(outdir, f"{self.setting_name}_{self.mode}_preds.pth"),
        )
        torch.save(
            targets, os.path.join(outdir, f"{self.setting_name}_{self.mode}_targs.pth")
        )

        self.test_pred.reset()


class Quantile(CatMetric):
    def __init__(
        self,
        q: float,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Any,
    ):
        super().__init__(nan_strategy=nan_strategy)
        self.q = q

    def compute(self) -> float:
        """Compute the aggregated value."""
        if isinstance(self.value, list) and self.value:
            self.value = dim_zero_cat(self.value)

        return torch.quantile(self.value, self.q).item()


class ExAD_AE_Module(ExAD_Module):
    def __init__(
        self,
        setting_name: str,
        image_size: int = 32,
        hidden_sizes: List[int] = [400, 40, 400],
        lr: float = 1e-4,
        outdir: str = "/exad_output",
    ):

        super().__init__(setting_name=setting_name, mode="AE", lr=lr, outdir=outdir)

        self.model = ExAD_AE(image_size=image_size, hidden_sizes=hidden_sizes)

        print(self.model)

        self.save_hyperparameters()


class ExAD_CNN_Module(ExAD_Module):
    def __init__(
        self, setting_name: str, lr: float = 1e-4, outdir: str = "/exad_output"
    ):

        super().__init__(setting_name=setting_name, mode="CNN", lr=lr, outdir=outdir)

        self.model = ExAD_CNN()

        self.save_hyperparameters()
