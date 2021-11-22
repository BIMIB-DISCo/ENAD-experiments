# Getting Started

Project of "EAD: an ensemble approach to detect adversarial examples from the hidden features of deep neural networks".

# Setup

The code was tested on `Python 3.7.10` with the packages listed in the `requirements.txt` file.

# Run Experiment

Run comparable experiments with the *Mahalanobis detector* (Maha) [[1]](#1).

1. Clone this repository in `path/to/repo`, and change the working directory:

    ```
    git clone https://github.com/... path/to/repo
    cd path/to/repo
    ```

2. Install the requirements (in a virtualenv):

    ```
    pip install -r requirements.txt
    ```

    Note: `scipy` and `scikit_learn` are not in the last version due to compatibility issues with the version of `scikit_optimize` used in the experiments.

3. Clone the Maha repository in the `mahalanobis` folder:

    ```
    git clone https://github.com/pokaxpoka/deep_Mahalanobis_detector.git mahalanobis
    ```

    <details>
    <summary>WARNING: compatibility with recent PyTorch versions.</summary>

    With recent PyTorch versions, a number of errors/warnings will show up and should be fixed, including:
    - replace the `volatile` flags with `with torch.no_grad()`.
    - replace `async=True` with `non_blocking=True` in `cuda`.
    - use `data` instead of `data[0]` for 0-dim tensors.
    - adding `.cpu()` before applying `.numpy()` to a tensor.
    </details>

4. Create the following folders:
    - `pre_trained`: it should contain the pre-trained models available [here](https://github.com/pokaxpoka/deep_Mahalanobis_detector.git).
    - `output`: it will contain the output of the experiments.

5. Generate the data for `cifar10`, `resnet` and `FGSM` (look [here](https://github.com/pokaxpoka/deep_Mahalanobis_detector.git) for the full code usage)

    ```
    python mahalanobis/ADV_Samples.py --dataset cifar10 --net_type resnet --adv_type FGSM --outf output/
    ```

    The command will generate four datasets in `output/resnet_cifar10`, where the *i*-th item is:
    - `clean_data_resnet_cifar10_FGSM.pth`: the benign (original) example *x* (correctly classified).
    - `adv_data_resnet_cifar10_FGSM.pth`: the adversarial example generated from *x* (misclassified).
    - `noisy_data_resnet_cifar10_FGSM.pth`: the noisy example generated from *x* (correctly classified).
    - `label_resnet_cifar10_FGSM.pth`: the target class of *x*.
    
    <details>
    <summary>Allowed parameter settings</summary>

    - `dataset`: `cifar10`, `cifar100`, `SVHN`;
    - `adv_type`: `FGSM`, `BIM`, `DeepFool`, `CWL2`,
    - `net_type`: `densenet`, `resnet` (if added to `pre_trained` folder).

    </details>

6. Generate the layer-specific scores of the Mahalanobis [[1]](#1) and LID [[2]](#2) detectors:

    ```
    python mahalanobis/ADV_Generate_LID_Mahalanobis.py --dataset cifar10 --net_type resnet --adv_type FGSM --outf output/
    ```

    The command generates for both the LID and Mahalanobis detector a table with the layer-specific scores in `LID_{parameter}_cifar10_FGSM.npy` and `Mahalanobis_{parameter}_cifar10_FGSM.npy`, where `{parameter}` is the number of neighbors *k* for LID and noise magnitude *lambda* for Mahalanobis. The tables are located in in `output/resnet_cifar10`.


7. Find the best hyper-parameters for the Mahalanobis and LID detectors:

    ```
    python maha_model_selection.py --dataset cifar10 --net_type resnet --adv_type FGSM --outf output
    ```

    The command generates a `LID_best_cifar10_FGSM.npy` and `Mahalanobis_best_cifar10_FGSM.npy` in `output/resnet_cifar10` with the tables of the layer-specific scores corresponding to the best parameters of LID and Mahalanobis.


9. Train the OCSVM detector:

    ```
    python ocsvm_trainer.py --dataset cifar10 --net_type resnet --adv_type FGSM --outf output
    ```

    The command generates in `output/resnet_cifar10/ocsvm` the following files:
    - `OCSVM_net_detector_resnet_cifar10_FGSM.pkl`: `net_detector` class containing the OCSVM detectors of each layer.
    - `OCSVM_resnet_cifar10_FGSM.npy`: OCSVM layer-specific scores on the test set.

    By default, it will use the best hyperparameters given in the paper and saved in `pre_computed/OCSVM_best_params.json`. When the `--precomputed` flag is set to `False`, use `--n_points` and `--n_iter` to choose the number of parallel evaluations and maximum number of iterations of the Bayesian hyperparameter optimization.

10. Train the Ensamble Adversarial Detector (EAD):

    ```
    python ead.py --dataset cifar10 --net_type resnet --adv_type FGSM --outf output
    ```

    Outputs the performances on the testset obtained by aggregating the (best) layer-specific scores of the LID, Mahalanobis and OCSVM detectors.

# Code Walkthrough

Apart from the files used above:
- `maha_extension` contains the code to reproduce the data splitting used in the Mahalanobis detector code (in `data.py`) and to use the pre-trained models (in `model.py`).
- `net_detector` contains the code to train a generic detector that extracts layer-specific scores and aggregates them using a logistic function:
    - `preprocessing.py` contains the `GroupedScaler` class to center data with the class means.
    - `net_detector.py` contains the class `net_detector`.
    - `base_detector.py` contains the Mahalanobis detector implemented using `GroupedScaler` (NOT USED).


# References

<a id="1">[1]</a>
Lee, Kimin, Kibok Lee, Honglak Lee, and Jinwoo Shin. 2018. “A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.” In Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, [paper](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html).



<a id="2">[2]</a>
Ma, Xingjun, Bo Li, Yisen Wang, Sarah M. Erfani, Sudanthi N. R. Wijewickrema, Grant Schoenebeck, Dawn Song, Michael E. Houle, and James Bailey. “Characterizing Adversarial Subspaces Using Local Intrinsic Dimensionality.” In 6th International Conference on Learning Representations, ICLR 2018, [paper](https://openreview.net/forum?id=B1gJ1L2aW).
