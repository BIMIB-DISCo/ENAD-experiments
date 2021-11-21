from mahalanobis import models

import torch
from torchvision import transforms

import numpy as np

def get_model_transforms(net_type, dataset, num_classes):

    pre_trained_net = f'pre_trained/{net_type}_{dataset}.pth'

    if net_type == 'densenet':
        if dataset == 'svhn':
            model = models.DenseNet3(100, num_classes)
            model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(0)))
        else:
            model = torch.load(pre_trained_net, map_location = "cuda:" + str(0))

        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),])

    elif net_type == 'resnet':

        # Model
        model = models.ResNet34(num_c=num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(0)))

        # Input Transforms
        in_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    else:
        raise ValueError()

    model.cuda()
    model.eval()

    return model, in_transform

def extract_activations(X, model, layer_idx, keep_grad=False, return_pred=False):

    if keep_grad:
        activations = model.intermediate_forward(X, layer_idx)
    else:
        with torch.no_grad():
            activations = model.intermediate_forward(X, layer_idx)

    activations = activations.view(activations.size(0), activations.size(1), -1)
    activations = torch.mean(activations, 2)

    if return_pred:
        output = model(X)
        y_pred = output.data.max(1)[1]

        return activations, y_pred

    else:

        return activations

def activations_from_loader(model, layer, loader):

    activations = []
    labels = []

    for x, y in loader:
        acts = extract_activations(x.cuda(), model, layer)

        activations.append(acts.cpu().numpy())
        labels.append(y.cpu().numpy())

    return np.concatenate(activations), np.concatenate(labels)