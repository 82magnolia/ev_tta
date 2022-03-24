from numpy.lib.shape_base import split
import torch
from math import ceil

"""
Collection of functions for converting (N, 4) event tensor to some other form
"""


def event_to_voxel(event: torch.Tensor, num_bins: int, num_events: int, height: int, width: int):
    """
    Convert (N, 4) event tensor into a new tensor of shape (N_e, B, H, W), where B is the number of bins to use,
    and N_e = ceil(N / num_events)

    Note that event = [x, y, time, polarity]

    Args:
        event: (N, 4) tensor containing events
        num_bins: Number of bins
        num_events: Unit number of events to pack to a single voxel batch of (B, H, W)
        height: Height of voxel
        width: Width of voxel
    
    Returns:
        voxel_event: (N_e, B, H, W) tensor containing voxelized events
    """
    tgt_event = event.clone().detach()
    # Swap x, y for indexing
    tgt_event = torch.index_select(tgt_event, 1, torch.LongTensor([1, 0, 2, 3]))
    N_e = ceil(tgt_event.shape[0] / num_events)
    voxel_event = torch.zeros([N_e, num_bins, height, width])
    
    for idx in range(N_e):
        sliced_event = tgt_event[num_events * (idx): num_events * (idx + 1)]
        time_step = sliced_event[-1, 2] - sliced_event[0, 2]
        # Normalize time
        sliced_event[:, 2] = num_bins * (sliced_event[:, 2] - sliced_event[0, 2]) / time_step

        floor_event = sliced_event.clone().detach()
        floor_event[:, 2] = torch.floor(sliced_event[:, 2])
        floor_event[:, 3] = sliced_event[:, 3] * (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))

        ceil_event = sliced_event.clone().detach()
        ceil_event[:, 2] = torch.ceil(sliced_event[:, 2])
        ceil_event[:, 3] = sliced_event[:, 3] * (1 - (torch.ceil(sliced_event[:, 2]) - sliced_event[:, 2]))

        dummy_bin_event = torch.cat([floor_event, ceil_event], dim=0)

        coords = dummy_bin_event[:, 0:3].long()
        new_coords = coords[coords[:, 2] < num_bins]
        val = dummy_bin_event[:, -1]
        val = val[coords[:, 2] < num_bins]

        bin_voxel_event = torch.sparse.FloatTensor(new_coords.t(), val, torch.Size([height, width, num_bins])).to_dense()
        bin_voxel_event = bin_voxel_event.permute(2, 0, 1)

        voxel_event[idx] = bin_voxel_event
    
    return voxel_event


def event_to_spike_tensor(event: torch.Tensor, num_bins: int, num_events: int, height: int, width: int, measure='time'):
    """
    Convert (N, 4) event tensor into a new tensor of shape (N_e, 2, B, H, W), where B is the number of bins to use,
    and N_e = ceil(N / num_events)

    Note that event = [x, y, time, polarity]

    Args:
        event: (N, 4) tensor containing events
        num_bins: Number of bins
        num_events: Unit number of events to pack to a single voxel batch of (B, H, W)
        height: Height of voxel
        width: Width of voxel
    
    Returns:
        voxel_event: (N_e, B, H, W) tensor containing voxelized events
    """
    assert measure in ['time', 'count', 'polarity', 'polarized_time']
    tgt_event = event.clone().detach()
    # Swap x, y for indexing
    tgt_event = torch.index_select(tgt_event, 1, torch.LongTensor([1, 0, 2, 3]))
    pos_event = tgt_event[tgt_event[:, -1] > 0]
    neg_event = tgt_event[tgt_event[:, -1] < 0]

    N_e = ceil(tgt_event.shape[0] / num_events)
    voxel_event = torch.zeros([N_e, 2, num_bins, height, width])
    
    for idx in range(N_e):
        # Positive Tensor
        sliced_event = pos_event[num_events * (idx): num_events * (idx + 1)]
        time_step = sliced_event[-1, 2] - sliced_event[0, 2]
        # Normalize time
        sliced_event[:, 2] = num_bins * (sliced_event[:, 2] - sliced_event[0, 2]) / time_step

        floor_event = sliced_event.clone().detach()
        floor_event[:, 2] = torch.floor(sliced_event[:, 2])
        
        if measure == 'polarity' or measure == 'count':
            floor_event[:, 3] = (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))
        elif measure == 'time' or measure == 'polarized_time':
            norm_time = sliced_event[:, 2] / num_bins
            floor_event[:, 3] = norm_time * (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))            

        ceil_event = sliced_event.clone().detach()
        ceil_event[:, 2] = torch.ceil(sliced_event[:, 2])

        if measure == 'polarity' or measure == 'count':
            ceil_event[:, 3] = (1 - (torch.ceil(sliced_event[:, 2]) - sliced_event[:, 2]))
        elif measure == 'time' or measure == 'polarized_time':
            norm_time = sliced_event[:, 2] / num_bins
            floor_event[:, 3] = norm_time * (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))

        dummy_bin_event = torch.cat([floor_event, ceil_event], dim=0)

        coords = dummy_bin_event[:, 0:3].long()
        new_coords = coords[coords[:, 2] < num_bins]
        val = dummy_bin_event[:, -1]
        val = val[coords[:, 2] < num_bins]

        bin_voxel_event = torch.sparse.FloatTensor(new_coords.t(), val, torch.Size([height, width, num_bins])).to_dense()
        bin_voxel_event = bin_voxel_event.permute(2, 0, 1)

        voxel_event[idx, 0] = bin_voxel_event

        # Negative Tensor
        sliced_event = neg_event[num_events * (idx): num_events * (idx + 1)]
        time_step = sliced_event[-1, 2] - sliced_event[0, 2]
        # Normalize time
        sliced_event[:, 2] = num_bins * (sliced_event[:, 2] - sliced_event[0, 2]) / time_step

        floor_event = sliced_event.clone().detach()
        floor_event[:, 2] = torch.floor(sliced_event[:, 2])
        
        if measure == 'polarity':
            floor_event[:, 3] = -(1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))
        elif measure == 'count':
            floor_event[:, 3] = (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))
        elif measure == 'time':
            norm_time = sliced_event[:, 2] / num_bins
            floor_event[:, 3] = norm_time * (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))
        elif measure == 'polarized_time':
            norm_time = sliced_event[:, 2] / num_bins
            floor_event[:, 3] = -norm_time * (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))

        ceil_event = sliced_event.clone().detach()
        ceil_event[:, 2] = torch.ceil(sliced_event[:, 2])

        if measure == 'polarity':
            ceil_event[:, 3] = -(1 - (torch.ceil(sliced_event[:, 2]) - sliced_event[:, 2]))
        elif measure == 'count':
            ceil_event[:, 3] = (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))
        elif measure == 'time':
            norm_time = sliced_event[:, 2] / num_bins
            ceil_event[:, 3] = norm_time * (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))
        elif measure == 'polarized_time':
            norm_time = sliced_event[:, 2] / num_bins
            floor_event[:, 3] = -norm_time * (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))

        dummy_bin_event = torch.cat([floor_event, ceil_event], dim=0)

        coords = dummy_bin_event[:, 0:3].long()
        new_coords = coords[coords[:, 2] < num_bins]
        val = dummy_bin_event[:, -1]
        val = val[coords[:, 2] < num_bins]

        bin_voxel_event = torch.sparse.FloatTensor(new_coords.t(), val, torch.Size([height, width, num_bins])).to_dense()
        bin_voxel_event = bin_voxel_event.permute(2, 0, 1)

        voxel_event[idx, 1] = bin_voxel_event
    
    return voxel_event


def event_to_voxel_full(event: torch.Tensor, num_bins, height, width, sparse=False):
    """
    Convert all the events to single voxel batch of shape (B, H, W)
    """
    if sparse:
        return event_to_voxel_sparse(event, num_bins, len(event), height, width)
    else:
        return event_to_voxel(event, num_bins, len(event), height, width).squeeze(0)


def event_to_spike_tensor_full(event: torch.Tensor, num_bins, height, width, measure='time'):
    """
    Convert all the events to single event spike tensor of shape (2, B, H, W)
    """
    return event_to_spike_tensor(event, num_bins, len(event), height, width, measure)


def event_to_count_voxel_full(event: torch.Tensor, num_bins, height, width):
    """
    Convert all the events to single event voxel of shape (B, H, W)
    """
    split_length = [event.shape[0] // num_bins] * num_bins if event.shape[0] % num_bins == 0 \
        else [event.shape[0] // num_bins] * (num_bins - 1) + [event.shape[0] // num_bins + event.shape[0] % num_bins]
    split_event = torch.split(event, split_length, dim=0)

    voxel_event = torch.zeros([num_bins, height, width])

    for idx, evt in enumerate(split_event):
        voxel_event[idx] = torch.bincount(evt[:, 0].long() + evt[:, 1].long() * width, minlength=height * width).reshape(height, width)

    return voxel_event, split_length


def event_to_voxel_sparse(event: torch.Tensor, num_bins: int, num_events: int, height: int, width: int):
    """
    Convert (N, 4) event tensor into a new tensor of shape (N_e, B, H, W), where B is the number of bins to use,
    and N_e = ceil(N / num_events). This methods returns the coordinates and values of the resulting tensor, instead
    of the tensor itself, for further use in sparse convolution.

    Note that event = [x, y, time, polarity]

    Args:
        event: (N, 4) tensor containing events
        num_bins: Number of bins
        num_events: Unit number of events to pack to a single voxel batch of (B, H, W)
        height: Height of voxel
        width: Width of voxel
    
    Returns:
        tot_coords: (N_tot, 4) tensor containing coordinates of the resulting tensor
        tot_vals: (N_tot, ) tensor containing values of the resulting tensor
    """
    tgt_event = event.clone().detach()
    # Swap x, y for indexing
    tgt_event = torch.index_select(tgt_event, 1, torch.LongTensor([1, 0, 2, 3]))
    N_e = ceil(tgt_event.shape[0] / num_events)
    tot_coords = []
    tot_vals = []
    
    for idx in range(N_e):
        sliced_event = tgt_event[num_events * (idx): num_events * (idx + 1)]
        time_step = sliced_event[-1, 2] - sliced_event[0, 2]
        # Normalize time
        sliced_event[:, 2] = num_bins * (sliced_event[:, 2] - sliced_event[0, 2]) / time_step

        floor_event = sliced_event.clone().detach()
        floor_event[:, 2] = torch.floor(sliced_event[:, 2])
        floor_event[:, 3] = sliced_event[:, 3] * (1 - (sliced_event[:, 2] - torch.floor(sliced_event[:, 2])))

        ceil_event = sliced_event.clone().detach()
        ceil_event[:, 2] = torch.ceil(sliced_event[:, 2])
        ceil_event[:, 3] = sliced_event[:, 3] * (1 - (torch.ceil(sliced_event[:, 2]) - sliced_event[:, 2]))

        dummy_bin_event = torch.cat([floor_event, ceil_event], dim=0)

        coords = dummy_bin_event[:, 0:3].long()
        new_coords = coords[coords[:, 2] != num_bins]
        val = dummy_bin_event[:, -1]
        val = val[coords[:, 2] != num_bins]

        tot_coords.append(new_coords)
        tot_vals.append(val)

    tot_coords = torch.cat(tot_coords, dim=0)
    tot_vals = torch.cat(tot_vals, dim=0)
    
    return tot_coords, tot_vals


if __name__ == '__main__':
    x = torch.load('/Datasets/imagenet_train_1278567.pt')

    z = event_to_voxel_full(x, 10, 224, 224, sparse=True)
    breakpoint()
