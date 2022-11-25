# Implementation of "ExAD: An Ensemble Approach for Explanation-based Adversarial Detection"

Code implementing the following work:

*"ExAD: An Ensemble Approach for Explanation-based Adversarial Detection"*.
Vardhan, Raj and Liu, Ninghao and Chinprutthiwong, Phakpoom and Fu, Weijie and Hu, Zhenyu and Hu, Xia Ben and Gu, Guofei. 2021. https://arxiv.org/abs/2103.11526.

with data generated through the following repo: https://github.com/pokaxpoka/deep_Mahalanobis_detector (`maha_repo`).

## Brief code walkthrough 

- `labeled_data` contains a `pytorch-lightning` `DataModule` defining training set and labeled set (with adv/benign labels) starting from the data generated through `maha_repo`.
- `attr_gen` contains the code to generate feature attributions from pre-trained models of `maha_repo`.
- `datamodule` defines a `pytorch-lightning` `DataModule` for attribution data used to trained *ExAD*.
- `load_model` contains the code to load pre-trained models of `maha_repo`.
- `models` defined *ExAD* detectors as `pytorch-lightning` `LightningModule`.
- `trainer` defines a `pytorch-lightning` `Trainer` to train *ExAD* detectors.

