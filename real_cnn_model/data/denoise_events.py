from numpy.core.arrayprint import _make_options_dict
from real_cnn_model.utils.convert_utils import event_to_count_voxel_full
import torch
import torch.nn.functional as F
from typing import Union

TIME_SCALE = 1000000


def density_filter(event_tensor: torch.Tensor, num_bins: int, height: int, width: int, neighborhood_size: int, threshold: int, neglect_polarity: bool = False):
    """
    Performs density filtering and outputs a filtered event_tensor.

    Args:
        event_tensor: (N, 4) event tensor
        num_bins: Number of bins to use for quantization
        height: Height of density matrix
        width: Width of density matrix
        neighborhood_size: Size of neighborhood
        threshold: Threshold value for deleting events
        neglect_polarity: If True, neglect polarity in denoising process
    
    Returns:
        event_tensor[mask]: Masked event_tensor
    """
    if neglect_polarity:
        vox_event, split_length = event_to_count_voxel_full(event_tensor, num_bins, height, width)
        if neighborhood_size > 1:
            vox_event = F.avg_pool2d(vox_event.unsqueeze(0), kernel_size=neighborhood_size, stride=1, padding=neighborhood_size // 2).squeeze(0)

        vox_indices = torch.zeros([event_tensor.shape[0], 3], dtype=torch.long)
        vox_indices[:, 0] = event_tensor[:, 1].long()  # i indices
        vox_indices[:, 1] = event_tensor[:, 0].long()  # j indices
        vox_indices[:, 2] = torch.repeat_interleave(torch.arange(num_bins), torch.LongTensor(split_length))

        keep_mtx = vox_event > threshold
        mask = keep_mtx[(vox_indices[:, 2], vox_indices[:, 0], vox_indices[:, 1])]
    
    else:
        pos_event = event_tensor[event_tensor[:, 3] > 0]
        neg_event = event_tensor[event_tensor[:, 3] < 0]

        mask = torch.zeros_like(event_tensor[:, -1], dtype=torch.bool)
        
        # pos events
        pos_vox_event, pos_split_length = event_to_count_voxel_full(pos_event, num_bins, height, width)
        if neighborhood_size > 1:
            pos_vox_event = F.avg_pool2d(pos_vox_event.unsqueeze(0), kernel_size=neighborhood_size, stride=1, padding=neighborhood_size // 2).squeeze(0)

        pos_vox_indices = torch.zeros([pos_event.shape[0], 3], dtype=torch.long)
        pos_vox_indices[:, 0] = pos_event[:, 1].long()  # i indices
        pos_vox_indices[:, 1] = pos_event[:, 0].long()  # j indices
        pos_vox_indices[:, 2] = torch.repeat_interleave(torch.arange(num_bins), torch.LongTensor(pos_split_length))

        pos_keep_mtx = pos_vox_event > threshold
        mask[event_tensor[:, 3] > 0] = pos_keep_mtx[(pos_vox_indices[:, 2], pos_vox_indices[:, 0], pos_vox_indices[:, 1])]

        # neg events
        neg_vox_event, neg_split_length = event_to_count_voxel_full(neg_event, num_bins, height, width)
        if neighborhood_size > 1:
            neg_vox_event = F.avg_pool2d(neg_vox_event.unsqueeze(0), kernel_size=neighborhood_size, stride=1, padding=neighborhood_size // 2).squeeze(0)

        neg_vox_indices = torch.zeros([neg_event.shape[0], 3], dtype=torch.long)
        neg_vox_indices[:, 0] = neg_event[:, 1].long()  # i indices
        neg_vox_indices[:, 1] = neg_event[:, 0].long()  # j indices
        neg_vox_indices[:, 2] = torch.repeat_interleave(torch.arange(num_bins), torch.LongTensor(neg_split_length))

        neg_keep_mtx = neg_vox_event > threshold
        mask[event_tensor[:, 3] < 0] = neg_keep_mtx[(neg_vox_indices[:, 2], neg_vox_indices[:, 0], neg_vox_indices[:, 1])]

    return event_tensor[mask]


def density_time_filter(event_tensor: torch.Tensor, timeslice: int, height: int, width: int, neighborhood_size: int, threshold: Union[int, float], 
        neglect_polarity: bool = False):
    """
    Performs density filtering and outputs a filtered event_tensor.

    Args:
        event_tensor: (N, 4) event tensor
        timeslice: Temporal slice to use for binning
        height: Height of density matrix
        width: Width of density matrix
        neighborhood_size: Size of neighborhood
        threshold: Threshold value for deleting events
        neglect_polarity: If True, neglect polarity in denoising process
    
    Returns:
        event_tensor[mask]: Masked event_tensor
    """
    if neglect_polarity:
        micro_time = ((event_tensor[:, 2] - event_tensor[0, 2]) * TIME_SCALE).long()
        split_length = torch.bincount(micro_time // timeslice).tolist()
        split_event = torch.split(event_tensor, split_length, dim=0)
        voxel_event = torch.zeros([len(split_length), height, width])

        for idx, evt in enumerate(split_event):
            voxel_event[idx] = torch.bincount(evt[:, 0].long() + evt[:, 1].long() * width, minlength=height * width).reshape(height, width)

        vox_indices = torch.zeros([event_tensor.shape[0], 3], dtype=torch.long)
        vox_indices[:, 0] = event_tensor[:, 1].long()  # i indices
        vox_indices[:, 1] = event_tensor[:, 0].long()  # j indices
        vox_indices[:, 2] = torch.repeat_interleave(torch.arange(len(split_length)), torch.LongTensor(split_length))

        if neighborhood_size > 1:
            voxel_event = F.avg_pool2d(voxel_event.unsqueeze(0), kernel_size=neighborhood_size, stride=1, padding=neighborhood_size // 2).squeeze(0)

        keep_mtx = voxel_event > threshold
        mask = keep_mtx[(vox_indices[:, 2], vox_indices[:, 0], vox_indices[:, 1])]
    
    else:
        pos_event = event_tensor[event_tensor[:, 3] > 0]
        neg_event = event_tensor[event_tensor[:, 3] < 0]

        mask = torch.zeros_like(event_tensor[:, -1], dtype=torch.bool)

        # pos events
        pos_micro_time = ((pos_event[:, 2] - pos_event[0, 2]) * TIME_SCALE).long()
        pos_split_length = torch.bincount(pos_micro_time // timeslice).tolist()
        pos_split_event = torch.split(pos_event, pos_split_length, dim=0)
        pos_voxel_event = torch.zeros([len(pos_split_length), height, width])

        for idx, evt in enumerate(pos_split_event):
            pos_voxel_event[idx] = torch.bincount(evt[:, 0].long() + evt[:, 1].long() * width, minlength=height * width).reshape(height, width)

        pos_vox_indices = torch.zeros([pos_event.shape[0], 3], dtype=torch.long)
        pos_vox_indices[:, 0] = pos_event[:, 1].long()  # i indices
        pos_vox_indices[:, 1] = pos_event[:, 0].long()  # j indices
        pos_vox_indices[:, 2] = torch.repeat_interleave(torch.arange(len(pos_split_length)), torch.LongTensor(pos_split_length))

        if neighborhood_size > 1:
            pos_voxel_event = F.avg_pool2d(pos_voxel_event.unsqueeze(0), kernel_size=neighborhood_size, stride=1, padding=neighborhood_size // 2).squeeze(0)

        keep_mtx = pos_voxel_event > threshold
        mask[event_tensor[:, 3] > 0] = keep_mtx[(pos_vox_indices[:, 2], pos_vox_indices[:, 0], pos_vox_indices[:, 1])]

        # neg events
        neg_micro_time = ((neg_event[:, 2] - neg_event[0, 2]) * TIME_SCALE).long()
        neg_split_length = torch.bincount(neg_micro_time // timeslice).tolist()
        neg_split_event = torch.split(neg_event, neg_split_length, dim=0)
        neg_voxel_event = torch.zeros([len(neg_split_length), height, width])

        for idx, evt in enumerate(neg_split_event):
            neg_voxel_event[idx] = torch.bincount(evt[:, 0].long() + evt[:, 1].long() * width, minlength=height * width).reshape(height, width)

        neg_vox_indices = torch.zeros([neg_event.shape[0], 3], dtype=torch.long)
        neg_vox_indices[:, 0] = neg_event[:, 1].long()  # i indices
        neg_vox_indices[:, 1] = neg_event[:, 0].long()  # j indices
        neg_vox_indices[:, 2] = torch.repeat_interleave(torch.arange(len(neg_split_length)), torch.LongTensor(neg_split_length))

        if neighborhood_size > 1:
            neg_voxel_event = F.avg_pool2d(neg_voxel_event.unsqueeze(0), kernel_size=neighborhood_size, stride=1, padding=neighborhood_size // 2).squeeze(0)

        keep_mtx = neg_voxel_event > threshold
        mask[event_tensor[:, 3] < 0] = keep_mtx[(neg_vox_indices[:, 2], neg_vox_indices[:, 0], neg_vox_indices[:, 1])]

    return event_tensor[mask]


def hot_pixel_filter(event_tensor: torch.Tensor, height: int, width: int, neglect_polarity: bool = False):
    """
    Performs hot pixel filtering and outputs a filtered event_tensor

    Args:
        event_tensor: (N, 4) event tensor
        height: Height of event image
        width: Width of event image
        neglect_polarity:  If True, neglect polarity in denoising process

    Returns:
        event_tensor[mask]: Masked event tensor
    """
    if neglect_polarity:
        event_img = torch.zeros([height, width])
        event_img[(event_tensor[:, 1].long(), event_tensor[:, 0].long())] = 1
        weight = torch.ones(3, 3)
        weight[1, 1] = 0
        weight = weight.reshape(1, 1, 3, 3)
        valid_img = (F.conv2d(event_img.reshape(1, 1, height, width), weight, padding=1).squeeze().long() > 0)

        mask = valid_img[(event_tensor[:, 1].long(), event_tensor[:, 0].long())]
    
    else:
        pos_event_img = torch.zeros([height, width])
        neg_event_img = torch.zeros([height, width])

        pos_event = event_tensor[event_tensor[:, 3] > 0]
        neg_event = event_tensor[event_tensor[:, 3] < 0]
        pos_event_img[(pos_event[:, 1].long(), pos_event[:, 0].long())] = 1
        neg_event_img[(neg_event[:, 1].long(), neg_event[:, 0].long())] = 1

        weight = torch.ones(3, 3)
        weight[1, 1] = 0
        weight = weight.reshape(1, 1, 3, 3)

        valid_img = (F.conv2d(pos_event_img.reshape(1, 1, height, width), weight, padding=1).squeeze().long() > 0) & \
            (F.conv2d(neg_event_img.reshape(1, 1, height, width), weight, padding=1).squeeze().long() > 0)
        
        mask = valid_img[(event_tensor[:, 1].long(), event_tensor[:, 0].long())]

    return event_tensor[mask]

