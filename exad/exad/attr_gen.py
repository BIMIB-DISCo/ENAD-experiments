import os
import warnings
from itertools import product
from typing import Any, Callable, Dict

import torch
from captum.attr import GuidedBackprop, InputXGradient, IntegratedGradients, NoiseTunnel
from jsonargparse import CLI
from tqdm.auto import tqdm

from .labeled_data import LabeledDataModule
from .load_model import load_model

warnings.filterwarnings("ignore")

ATTR_PARAMS = {
    "InputXGradient": {"batch_size": 20},
    "GuidedBackprop": {"batch_size": 20},
    "IntegratedGradients": {
        "batch_size": 1,
        "attr_args": {
            "method": "gausslegendre",
            "baselines": None,
            "n_steps": 100,
        },
    },
}
MODEL_NAMES = ["densenet", "resnet"]
DATASET_NAMES = ["cifar10", "svhn"]
ATTACK_NAMES = ["FGSM", "BIM", "DeepFool", "CWL2"]


class GenerateAttributions:
    def __init__(self, attr_name: str):

        if attr_name == "IntegratedGradients":
            self.attr_cls = IntegratedGradients
        elif attr_name == "InputXGradient":
            self.attr_cls = InputXGradient
        elif attr_name == "GuidedBackprop":
            self.attr_cls = GuidedBackprop
        else:
            raise ValueError()

    def generate_attributions(
        self,
        model: Callable,
        datamodule: LabeledDataModule,
        model_name: str,
        dataset_name: str,
        attack_name: str,
        attr_name: str,
        out_dirpath: str,
        batch_size: int = 100,
        attr_args: Dict[str, Any] = {},
        force_overwrite: bool = False,
        nt_stdevs: float = 0.2,
        nt_type: str = "smoothgrad_sq",
        nt_samples: int = 10,
    ):

        if attr_name == "GuidedBackprop" and model_name == "resnet":
            return

        noise_tunnel = NoiseTunnel(self.attr_cls(model))

        dataloaders = {
            "train": datamodule.train_dataloader(batch_size=batch_size),
            "ltrain": datamodule.labeled_train_dataloader(batch_size=batch_size),
            "lval": datamodule.labeled_val_dataloader(batch_size=batch_size),
            "ltest": datamodule.labeled_test_dataloader(batch_size=batch_size),
        }

        for dl_name, dl in dataloaders.items():

            if dl_name != "train":
                setting_name = f"{model_name}_{dataset_name}_{attack_name}_{attr_name}"
            else:
                setting_name = f"{model_name}_{dataset_name}_{attr_name}"

            data_fname = f"{setting_name}_{dl_name}.pth"
            target_fname = f"{setting_name}_{dl_name}_targets.pth"

            if (
                os.path.isfile(os.path.join(out_dirpath, data_fname))
                and not force_overwrite
            ):
                continue

            attributions = []
            pred_labels_idxs = []

            for x, y in tqdm(dl, leave=False):
                pred_label_idx = model(x.cuda()).argmax(1)

                x_attr = (
                    noise_tunnel.attribute(
                        x.cuda(),
                        nt_samples=nt_samples,
                        nt_samples_batch_size=1,
                        nt_type=nt_type,
                        target=pred_label_idx,
                        stdevs=nt_stdevs,
                        **attr_args,
                    )
                    .detach()
                    .cpu()
                )

                attributions.append(x_attr)
                pred_labels_idxs.append(pred_label_idx.cpu())

            attributions = torch.cat(attributions).cpu()
            pred_labels_idxs = torch.cat(pred_labels_idxs).cpu()

            out_fpath = os.path.join(out_dirpath, data_fname)
            torch.save(attributions, out_fpath)

            out_fpath = os.path.join(out_dirpath, target_fname)
            torch.save(pred_labels_idxs, out_fpath)


def generate_all(out_dir: str, maha_repo_dir: str, pretrained_dir: str):
    """Generate all the attributions into the out_dir folder. The pretrained models
    are from https://github.com/pokaxpoka/deep_Mahalanobis_detector (maha_repo).

    Args:
        out_dir (str): output dir.
        maha_repo_dir (str): path to maha_repo.
        pretrained_dir (str): location of pretrained models from maha_repo.
    """

    all_settings = list(product(MODEL_NAMES, DATASET_NAMES, ATTACK_NAMES))

    for attr_name, attr_params in tqdm(ATTR_PARAMS.items()):

        attr = GenerateAttributions(attr_name=attr_name)

        pbar = tqdm(all_settings)

        for model_name, dataset_name, attack_name in pbar:

            pbar.set_description(
                f"{model_name}_{dataset_name}_{attack_name}_{attr_name}"
            )

            datamodule = LabeledDataModule(model_name, dataset_name, attack_name)
            datamodule.setup()
            model = (
                load_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    maha_repo_dir=maha_repo_dir,
                    pretrained_dir=pretrained_dir,
                )
                .cuda()
                .eval()
            )

            attr.generate_attributions(
                model=model,
                datamodule=datamodule,
                model_name=model_name,
                dataset_name=dataset_name,
                attack_name=attack_name,
                attr_name=attr_name,
                out_dirpath=out_dir,
                **attr_params,
            )


if __name__ == "__main__":
    CLI(generate_all)
