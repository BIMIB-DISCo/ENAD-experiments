ALL_NET_TYPES = ['resnet', 'densenet']
ALL_DATASETS = ['cifar10', 'svhn', 'cifar100']
ALL_ADV_TYPES = ['DeepFool', 'FGSM', 'BIM', 'CWL2']

n_classes = lambda dataset: 100 if dataset == 'cifar100' else 10
n_layers = lambda net_type: 5 if net_type == 'resnet' else 4