import numpy as np
import torch

from .model import extract_activations, activations_from_loader

from mahalanobis import data_loader

class Datasets():

    def __init__(self, ds_name, in_transform, net_type, adv_type, outf):

        self.ds_name = ds_name
        self.net_type = net_type
        self.adv_type = adv_type

        self.train_loader, _ = data_loader.getTargetDataSet(ds_name, 100, in_transform, './data')

        fname = f"{net_type}_{ds_name}_{adv_type}.pth"

        test_data = torch.load(f'{outf}/clean_data_{fname}')
        new_size = 100 * (len(test_data) // 100)
        test_data = test_data[:new_size]

        noisy_data = torch.load(f'{outf}/noisy_data_{fname}')[:new_size]
        adv_data = torch.load(f'{outf}/adv_data_{fname}')[:new_size]
        targets = torch.load(f'{outf}/label_{fname}').cpu().numpy()[:new_size]

        self.X_test = torch.cat([adv_data, test_data, noisy_data]).cuda()
        self.y_test = np.tile(targets, 3)
        self.adv_test = np.array([-1] * (len(adv_data)) + [1] * (len(test_data) + len(noisy_data)))

        # Splits
        p_size = len(test_data)
        p_split = int(p_size*0.1)
        idxs_trainval = np.concatenate([
            np.arange(p_split),
            np.arange(p_size, p_size+p_split),
            np.arange(2*p_size, 2*p_size+p_split)
        ])

        self.idxs_test = np.delete(np.arange(len(self.X_test)), idxs_trainval)

        pivot = int(len(idxs_trainval) / 6)
        self.idxs_train = np.concatenate([idxs_trainval[:pivot], idxs_trainval[2*pivot:3*pivot], idxs_trainval[4*pivot:5*pivot]])
        self.idxs_val = np.concatenate([idxs_trainval[pivot:2*pivot], idxs_trainval[3*pivot:4*pivot], idxs_trainval[5*pivot:]])

class TrainValLoader:

    def __init__(self, model, ds, batch_size=100):
        self.ds = ds
        self.model = model
        self.batch_size = batch_size

    def __call__(self, layer_idx):

        X_train, y_train = activations_from_loader(self.model, layer_idx, self.ds.train_loader)
        adv_train = np.repeat(1, len(X_train))

        X_valid = self.ds.X_test[self.ds.idxs_val]
        acts_valid, y_valid = [], []
        for batch in torch.split(X_valid, self.batch_size):
            a_temp, y_temp = extract_activations(batch, self.model, layer_idx, return_pred=True)
            acts_valid.append(a_temp.cpu().numpy())
            y_valid.append(y_temp.cpu().numpy())
        X_valid = np.concatenate(acts_valid)
        y_valid = np.concatenate(y_valid)
        adv_valid = self.ds.adv_test[self.ds.idxs_val]

        return X_train, X_valid, y_train, y_valid, adv_train, adv_valid


class LabelledTrainLoader:

    def __init__(self, model, ds, batch_size=100):
        self.ds = ds
        self.model = model
        self.batch_size = batch_size

    def __call__(self, layer_idx):
        X_test = self.ds.X_test[self.ds.idxs_train]

        acts, y_test = [], []
        for batch in torch.split(X_test, self.batch_size):
            a_temp, y_temp = extract_activations(batch, self.model, layer_idx, return_pred=True)
            acts.append(a_temp.cpu().numpy())
            y_test.append(y_temp.cpu().numpy())
        acts = np.concatenate(acts)
        y_test = np.concatenate(y_test)

        return acts, y_test

class LabelledTestLoader:

    def __init__(self, model, ds, batch_size=100):
        self.ds = ds
        self.model = model
        self.batch_size = batch_size

    def __call__(self, layer_idx):
        X_test = self.ds.X_test[self.ds.idxs_test]

        acts, y_test = [], []
        for batch in torch.split(X_test, self.batch_size):
            a_temp, y_temp = extract_activations(batch, self.model, layer_idx, return_pred=True)
            acts.append(a_temp.cpu().numpy())
            y_test.append(y_temp.cpu().numpy())
        acts = np.concatenate(acts)
        y_test = np.concatenate(y_test)

        return acts, y_test

def idxs_train_val_test(data_size):
    p_size = data_size // 3
    p_split = int(p_size*0.1)
    idxs_trainval = np.concatenate([
        np.arange(p_split),
        np.arange(p_size, p_size+p_split),
        np.arange(2*p_size, 2*p_size+p_split)
       ])

    idxs_test = np.delete(np.arange(data_size), idxs_trainval)

    pivot = int(len(idxs_trainval) / 6)
    idxs_train = np.concatenate([idxs_trainval[:pivot], idxs_trainval[2*pivot:3*pivot], idxs_trainval[4*pivot:5*pivot]])
    idxs_val = np.concatenate([idxs_trainval[pivot:2*pivot], idxs_trainval[3*pivot:4*pivot], idxs_trainval[5*pivot:]])

    return idxs_train, idxs_val, idxs_test