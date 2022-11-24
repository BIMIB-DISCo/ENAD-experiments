import os
import sys
import torch


def load_model(
    model_name: str,
    dataset_name: str,
    maha_repo_dir: str,
    pretrained_dir: str,
    num_classes: int = 10,
):
    """
    Let maha_repo be https://github.com/pokaxpoka/deep_Mahalanobis_detector.git.

    Args:
        model_name (str): either densenet or resnet.
        dataset_name (str): either cifar10, cifar100 or svhn.
        maha_repo_dir (str): location of maha_repo.
        pretrained_dir (str): location of pre-trained models of maha_repo.
        num_classes (int, optional): Defaults to 10.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """

    sys.path.append(maha_repo_dir)

    model_path = os.path.join(pretrained_dir, f"{model_name}_{dataset_name}.pth")

    if model_name == "densenet":
        if dataset_name == "svhn":
            from mahalanobis import models  # type: ignore

            model = models.DenseNet3(100, num_classes)
            model.load_state_dict(torch.load(model_path))
        else:

            model = torch.load(f"{model_path}")

    elif model_name == "resnet":
        from mahalanobis import models  # type: ignore

        model = models.ResNet34(num_c=num_classes)
        model.load_state_dict(torch.load(model_path))

    else:
        raise ValueError()

    return model
