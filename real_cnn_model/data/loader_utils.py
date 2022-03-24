from real_cnn_model.data.imagenet import reshape_then_acc_adj_sort, reshape_then_acc_count_pol, reshape_then_acc_time_pol, \
    reshape_then_acc_exp, reshape_then_acc_count, reshape_then_acc_sort, reshape_then_flat_pol, HATS, reshape_then_flat
import torch
import random


def get_loader(loader_name: str):
    # Choose loader ((N, 4) event tensor -> Network input)
    loader_name = loader_name.replace('reshape_then_', '')

    if loader_name == 'acc_count':
        loader = reshape_then_acc_count
    elif loader_name == 'flat':
        loader = reshape_then_flat
    elif loader_name == 'flat_pol':
        loader = reshape_then_flat_pol
    elif loader_name == 'acc_time_pol':
        loader = reshape_then_acc_time_pol
    elif loader_name == 'acc_count_pol':
        loader = reshape_then_acc_count_pol
    elif loader_name == 'acc_exp':
        loader = reshape_then_acc_exp
    elif loader_name == 'acc_sort':
        loader = reshape_then_acc_sort
    elif loader_name == 'acc_adj_sort':
        loader = reshape_then_acc_adj_sort
    elif loader_name == 'HATS':
        loader = HATS

    return loader


class ComposableLoader:
    """
    Concatenates multiple representations into a single event representation.
    """
    def __init__(self, loader_list: list):
        self.loader_list = [get_loader(loader_name) for loader_name in loader_list]
    
    def __call__(self, event_tensor, augment=None, **kwargs):
        result = [representation(event_tensor, augment, **kwargs) for representation in self.loader_list]
        # Each representation is C * H * W
        return torch.cat(result, dim=0)


class MixedLoader:
    """
    Mixes multiple representations, please provide a set of loaders with the same number of channels.
    Each representation provided is called randomly and fed into the classifier network.
    """
    def __init__(self, loader_list):
        self.loader_list = [get_loader(loader_name) for loader_name in loader_list]
    
    def __call__(self, event_tensor, augment=None, **kwargs):
        representation = random.choice(self.loader_list)
        # Each representation is C * H * W
        return representation(event_tensor, augment=None, **kwargs)
