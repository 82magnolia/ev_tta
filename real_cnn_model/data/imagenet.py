import torch
from torch_scatter import scatter_max, scatter_min
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
import torch.nn.functional as F
from scipy.ndimage import median_filter, laplace
from queue import Queue
from real_cnn_model.data.denoise_events import density_filter, hot_pixel_filter, density_time_filter


SENSOR_H = 480
SENSOR_W = 640
IMAGE_H = 224
IMAGE_W = 224
EXP_TAU = 0.3
TIME_SCALE = 1000000
MAX_TRAJECTORY_SPEED = 0.0  # Speed per second limit
MAHALANOBIS = 0.0  # Threshold for removing noise
DENSITY_SIZE = 3
FLASH_DENSITY_SIZE = 13
FLASH_TH = 0.7  # Threshold for flashes

CLIP_COUNT = False
CLIP_COUNT_RATE = 0.99
DISC_ALPHA = 3.0

# Parsing Modules


def load_event(event_path, cfg):
    # Returns time-shifted numpy array event from event_path
    event = np.load(event_path)
    if getattr(cfg, 'compressed', True):
        event = event['event_data']
        event = np.vstack([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T
    else:
        event = np.vstack([event['x_pos'], event['y_pos'], event['timestamp'], event['polarity'].astype(np.uint8)]).T

    event = event.astype(float)

    # Account for int-type timestamp
    event[:, 2] /= TIME_SCALE

    # Account for zero polarity
    if event[:, 3].min() >= -0.5:
        event[:, 3][event[:, 3] <= 0.5] = -1

    return event


def slice_event(event, cfg):
    slice_method = getattr(cfg, 'slice_method', 'idx')
    if slice_method == 'idx':
        start = getattr(cfg, 'slice_start', None)
        end = getattr(cfg, 'slice_end', None)
        event = event[start:end]
    elif slice_method == 'time':
        start = getattr(cfg, 'slice_start', None)
        end = getattr(cfg, 'slice_end', None)
        event = event[(event[:, 2] > start) & (event[:, 2] < end)]
    elif slice_method == 'random':
        length = getattr(cfg, 'slice_length', None)
        slice_augment = getattr(cfg, 'slice_augment', False)
        if slice_augment and cfg.mode == 'train':
            slice_augment_width = getattr(cfg, 'slice_augment_width', 0)
            slice_augment_width_l = getattr(cfg, 'slice_augment_width_l', 0)
            slice_augment_width_r = getattr(cfg, 'slice_augment_width_r', 0)
            if slice_augment_width != 0:
                length = random.randint(length - slice_augment_width, length + slice_augment_width)
            else:
                length = random.randint(length - slice_augment_width_l, length + slice_augment_width_r)
        if len(event) > length:
            start = random.choice(range(len(event) - length + 1))
            event = event[start: start + length]

    return event


def reshape_event_with_sample(event, orig_h, orig_w, new_h, new_w):
    # Sample events
    sampling_ratio = (new_h * new_w) / (orig_h * orig_w)

    new_size = int(sampling_ratio * len(event))
    idx_arr = np.arange(len(event))

    sampled_arr = np.random.choice(idx_arr, size=new_size, replace=False)
    sampled_event = event[np.sort(sampled_arr)]

    # Rescale coordinates
    sampled_event[:, 0] *= (new_w / orig_w)
    sampled_event[:, 1] *= (new_h / orig_h)

    return sampled_event


def reshape_event_no_sample(event, orig_h, orig_w, new_h, new_w):
    event[:, 0] *= (new_w / orig_w)
    event[:, 1] *= (new_h / orig_h)

    return event


def reshape_event_unique(event, orig_h, orig_w, new_h, new_w):
    event[:, 0] *= (new_w / orig_w)
    event[:, 1] *= (new_h / orig_h)

    coords = event[:, :2].astype(np.int64)
    timestamp = (event[:, 2] * TIME_SCALE).astype(np.int64)
    min_time = timestamp[0]
    timestamp -= min_time

    key = coords[:, 0] + coords[:, 1] * new_w + timestamp * new_h * new_w
    _, unique_idx = np.unique(key, return_index=True)

    event = event[unique_idx]

    return event


def parse_event(event_path, cfg):
    event = load_event(event_path, cfg)
    
    event = torch.from_numpy(event)

    # Account for density-based denoising
    denoise_events = getattr(cfg, 'denoise_events', False)
    denoise_bins = getattr(cfg, 'denoise_bins', 10)
    denoise_timeslice = getattr(cfg, 'denoise_timeslice', 5000)
    denoise_patch = getattr(cfg, 'denoise_patch', 2)
    denoise_thres = getattr(cfg, 'denoise_thres', 0.5)
    denoise_density = getattr(cfg, 'denoise_density', False)
    denoise_hot = getattr(cfg, 'denoise_hot', False)
    denoise_time = getattr(cfg, 'denoise_time', False)
    denoise_neglect_polarity = getattr(cfg, 'denoise_neglect_polarity', True)

    reshape = getattr(cfg, 'reshape', False)
    if reshape:
        reshape_method = getattr(cfg, 'reshape_method', 'no_sample')

        if reshape_method == 'no_sample':
            event = reshape_event_no_sample(event, SENSOR_H, SENSOR_W, IMAGE_H, IMAGE_W)
        elif reshape_method == 'sample':
            event = reshape_event_with_sample(event, SENSOR_H, SENSOR_W, IMAGE_H, IMAGE_W)
        elif reshape_method == 'unique':
            event = reshape_event_unique(event, SENSOR_H, SENSOR_W, IMAGE_H, IMAGE_W)

    if denoise_events:
        assert not (denoise_density and denoise_time) 
        if denoise_density:
            event = density_filter(event, num_bins=denoise_bins, height=IMAGE_H, width=IMAGE_W, 
                neighborhood_size=denoise_patch, threshold=denoise_thres, neglect_polarity=denoise_neglect_polarity)
        if denoise_time:
            event = density_time_filter(event, timeslice=denoise_timeslice, height=IMAGE_H, width=IMAGE_W, 
                neighborhood_size=denoise_patch, threshold=denoise_thres, neglect_polarity=denoise_neglect_polarity)

    # Account for slicing
    slice_events = getattr(cfg, 'slice_events', False)

    if slice_events:
        event = slice_event(event, cfg)

    if denoise_events:
        if denoise_hot:
            event = hot_pixel_filter(event, height=IMAGE_H, width=IMAGE_W, neglect_polarity=denoise_neglect_polarity)

    return event


def tree_parse_event(event_path, cfg):
    def event2img(event_tensor, reshape):
        if reshape:
            coords = event_tensor[:, :2].long()
            event_image = torch.zeros([IMAGE_H, IMAGE_W])
            event_image[(coords[:, 1], coords[:, 0])] = 1.0
        else:
            coords = event_tensor[:, :2].long()
            event_image = torch.zeros([SENSOR_H, SENSOR_W])
            event_image[(coords[:, 1], coords[:, 0])] = 1.0

        return event_image

    def event2count(event_tensor, reshape):
        assert reshape
        event_img = reshape_then_acc_count_only(event_tensor).squeeze()

        return event_img

    def metric(event_img, **kwargs):
        # Evaluate event_img
        # Higher the metric, the better! (At least normally..)
        metric_list = cfg.metric_list
        total_metric = 0.0
        if 'gradient' in metric_list:
            gy, gx = np.gradient(event_img.numpy())
            gnorm = np.sqrt(gx**2 + gy**2)
            sharpness = np.average(gnorm)
            total_metric += sharpness

        if 'variance' in metric_list:
            total_metric += torch.var(event_img).item()

        if 'laplace' in metric_list:
            total_metric += (np.average(np.abs(laplace(event_img.numpy()))))
        
        return total_metric

    event = load_event(event_path, cfg)
    reshape = getattr(cfg, 'reshape', False)
    projection_type = getattr(cfg, 'projection_type', 'event2img')

    if projection_type == 'event2img':
        projection_func = event2img
    elif projection_type == 'event2count':
        projection_func = event2count

    if reshape:
        reshape_method = getattr(cfg, 'reshape_method', 'no_sample')

        if reshape_method == 'no_sample':
            event = reshape_event_no_sample(event, SENSOR_H, SENSOR_W, IMAGE_H, IMAGE_W)
        elif reshape_method == 'sample':
            event = reshape_event_with_sample(event, SENSOR_H, SENSOR_W, IMAGE_H, IMAGE_W)
        elif reshape_method == 'unique':
            event = reshape_event_unique(event, SENSOR_H, SENSOR_W, IMAGE_H, IMAGE_W)

    # Use tree search to find optimal index to start indexing
    slice_size = getattr(cfg, 'tree_slice_size', 30000)
    num_split = max(len(event) // slice_size, 1) if cfg.num_split == 'adaptive' else cfg.num_split

    max_depth = cfg.max_depth

    total_event = torch.from_numpy(event)
    event_len = len(event)

    event_list = Queue()
    event_list.put(0)
    reverse_metric = getattr(cfg, 'reverse_metric', False)

    if slice_size > len(event):
        return total_event

    for d in range(max_depth):
        orig_idx = event_list.get()
        orig_img = projection_func(total_event[orig_idx: orig_idx + slice_size], reshape)
        orig_metric = metric(orig_img, event=total_event[orig_idx: orig_idx + slice_size], global_event=total_event)

        # Split event into num_split tensors
        split_event_list = [(orig_idx + event_len // num_split * i, metric(projection_func(
            total_event[orig_idx + event_len // num_split * i:orig_idx + event_len // num_split * i + slice_size],
            reshape), event=total_event[orig_idx + event_len // num_split * i:orig_idx + event_len // num_split * i + slice_size],
            global_event=total_event)) for i in range(num_split)]

        # Filter out of region events and it all event slices are out of range break
        split_event_list = list(filter(lambda x: x[0] + slice_size - 1 < len(event), split_event_list))

        if len(split_event_list) == 0:
            event_list.put(orig_idx)
            break

        split_event_list = sorted(split_event_list, key=lambda x: x[1], reverse=not reverse_metric)

        best_idx, best_split_metric = split_event_list[0]

        if reverse_metric:
            if best_split_metric >= orig_metric:
                event_list.put(orig_idx)
                break

            else:
                event_list.put(best_idx)
        else:
            if best_split_metric <= orig_metric:
                event_list.put(orig_idx)
                break

            else:
                event_list.put(best_idx)

        event_len = event_len // num_split
    idx = event_list.get()

    return total_event[idx: idx + slice_size]

# Aggregation Modules

def reshape_then_acc_time(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 4 * H * W image (Extended Timestamp Image)

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)

    pos_min_out, _ = scatter_min(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_min_out = pos_min_out.reshape(H, W)
    neg_min_out, _ = scatter_min(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_min_out = neg_min_out.reshape(H, W)

    result = torch.stack([pos_min_out, pos_out, neg_min_out, neg_out], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_count(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 4 * H * W image (Event Image)

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    # Account for empty events
    if len(event_tensor) == 0:
        event_tensor = torch.zeros([10, 4]).float()
        event_tensor[:, 2] = torch.arange(10) / 10.
        event_tensor[:, -1] = 1

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)

    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)
    result = torch.stack([pos_count, pos_out, neg_count, neg_out], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_diff_dist(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 4 * H * W image (Data for differentiable DiST)
    # Returns newest timestamp + inverted event count

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    # Account for empty events
    if len(event_tensor) == 0:
        event_tensor = torch.zeros([10, 4]).float()
        event_tensor[:, 2] = torch.arange(10) / 10.
        event_tensor[:, -1] = 1

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W).float()
    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W).float()

    # Neighborhood aggregation
    patch_size = 5

    avg_pos_count = patch_size ** 2 * torch.nn.functional.avg_pool2d(pos_count.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)
    avg_neg_count = patch_size ** 2 * torch.nn.functional.avg_pool2d(neg_count.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)

    avg_pos_count = avg_pos_count.squeeze(0)
    avg_neg_count = avg_neg_count.squeeze(0)

    avg_pos_count[pos_count == 0] = 0
    avg_neg_count[neg_count == 0] = 0
    avg_pos_count[pos_count > 0] = 1 / avg_pos_count[pos_count > 0]
    avg_neg_count[neg_count > 0] = 1 / avg_neg_count[neg_count > 0]

    pos_count = avg_pos_count
    neg_count = avg_neg_count

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)
    result = torch.stack([pos_count, pos_out, neg_count, neg_out], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_count_pol(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image (Event Histogram)

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)
    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)

    result = torch.stack([pos_count, neg_count], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_count_only(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 1 * H * W image (Compressed Event Histogram)

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    # Get pos, neg counts
    event_count = torch.bincount(event_tensor[:, 0].long() + event_tensor[:, 1].long() * W, minlength=H * W).reshape(H, W)

    result = torch.unsqueeze(event_count, -1)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_flat(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 1 * H * W image (Compressed Binary Event Image)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    if augment is not None:
        event_tensor = augment(event_tensor)

    coords = event_tensor[:, :2].long()
    event_image = torch.zeros([H, W])
    event_image[(coords[:, 1], coords[:, 0])] = 1.0

    event_image = torch.unsqueeze(event_image, -1)

    event_image = event_image.permute(2, 0, 1)
    event_image = event_image.float()

    return event_image


def reshape_then_flat_pol(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image (Binary Event Image)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    if augment is not None:
        event_tensor = augment(event_tensor)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    pos_coords = pos[:, :2].long()
    neg_coords = neg[:, :2].long()

    pos_image = torch.zeros([H, W])
    neg_image = torch.zeros([H, W])

    pos_image[(pos_coords[:, 1], pos_coords[:, 0])] = 1.0
    neg_image[(neg_coords[:, 1], neg_coords[:, 0])] = 1.0

    event_image = torch.stack([pos_image, neg_image], dim=2)
    event_image = event_image.permute(2, 0, 1)
    event_image = event_image.float()

    return event_image


def reshape_then_acc_exp(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image (Time Surface)

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    pos_out_exp = torch.exp(-(1 - pos_out) / EXP_TAU)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)
    neg_out_exp = torch.exp(-(1 - neg_out) / EXP_TAU)

    result = torch.stack([pos_out_exp, neg_out_exp], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_time_pol(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image (Timestamp Image)

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    # Account for empty events
    if len(event_tensor) == 0:
        event_tensor = torch.zeros([10, 4]).float()
        event_tensor[:, 2] = torch.arange(10) / 10.
        event_tensor[:, -1] = 1

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg time
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)

    result = torch.stack([pos_out, neg_out], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result


def reshape_then_acc_sort(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image (Sorted Time Surface)
    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    # Get sorted indices of event tensor
    time_idx = (event_tensor[:, 2] * TIME_SCALE).long()
    pos_time_idx = time_idx[event_tensor[:, 3] > 0]
    neg_time_idx = time_idx[event_tensor[:, 3] < 0]

    pos_mem, pos_cnt = torch.unique_consecutive(pos_time_idx, return_counts=True)
    pos_time_idx = torch.repeat_interleave(torch.arange(pos_mem.shape[0]), pos_cnt)

    neg_mem, neg_cnt = torch.unique_consecutive(neg_time_idx, return_counts=True)
    neg_time_idx = torch.repeat_interleave(torch.arange(neg_mem.shape[0]), neg_cnt)
    
    event_tensor[:, 2] = time_idx

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]

    if pos.shape[0] == 0:
        pos = torch.zeros(1, 4)
        pos[:, -1] = 1
    if neg.shape[0] == 0:
        neg = torch.zeros(1, 4)
        neg[:, -1] = 1

    # Get pos sort
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    pos_scatter_result, pos_scatter_idx = scatter_max(pos[:, 2], pos_idx, dim=-1, dim_size=H * W)

    pos_idx_mask = torch.zeros(pos.shape[0], dtype=torch.bool)
    pos_idx_mask[pos_scatter_idx[pos_scatter_idx < pos_idx.shape[0]]] = True
    tmp_pos = pos[pos_idx_mask]

    pos_final_mem, pos_final_cnt = torch.unique_consecutive(tmp_pos[:, 2], return_counts=True)
    # One is added to ensure that sorted values are greater than 1
    pos_final_scatter = torch.repeat_interleave(torch.arange(pos_final_mem.shape[0]), pos_final_cnt).float() + 1

    if pos_final_scatter.max() != pos_final_scatter.min():
        pos_final_scatter = (pos_final_scatter - pos_final_scatter.min()) / (pos_final_scatter.max() - pos_final_scatter.min())
    else:
        pos_final_scatter.fill_(0.0)
    
    pos_sort = torch.zeros(H, W)
    pos_coords = tmp_pos[:, :2].long()

    pos_sort[(pos_coords[:, 1], pos_coords[:, 0])] = pos_final_scatter
    pos_sort = pos_sort.reshape(H, W)

    # Get neg_sort
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    neg_scatter_result, neg_scatter_idx = scatter_max(neg[:, 2], neg_idx, dim=-1, dim_size=H * W)

    neg_idx_mask = torch.zeros(neg.shape[0], dtype=torch.bool)
    neg_idx_mask[neg_scatter_idx[neg_scatter_idx < neg_idx.shape[0]]] = True
    tmp_neg = neg[neg_idx_mask]

    neg_final_mem, neg_final_cnt = torch.unique_consecutive(tmp_neg[:, 2], return_counts=True)
    # One is added to ensure that sorted values are greater than 1
    neg_final_scatter = torch.repeat_interleave(torch.arange(neg_final_mem.shape[0]), neg_final_cnt).float() + 1
    if neg_final_scatter.max() != neg_final_scatter.min():
        neg_final_scatter = (neg_final_scatter - neg_final_scatter.min()) / (neg_final_scatter.max() - neg_final_scatter.min())
    else:
        neg_final_scatter.fill_(0.0)

    neg_sort = torch.zeros(H, W)
    neg_coords = tmp_neg[:, :2].long()

    neg_sort[(neg_coords[:, 1], neg_coords[:, 0])] = neg_final_scatter
    neg_sort = neg_sort.reshape(H, W)
    
    result = torch.stack([pos_sort, neg_sort], dim=2)
    result = result.permute(2, 0, 1)
    result = result.float()

    return result


def reshape_then_acc_intensity(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 1 * H * W image (Event Intensity Image)

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)
    pos_count = pos_count.float()

    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)
    neg_count = neg_count.float()

    intensity = pos_count - neg_count
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())

    result = intensity.unsqueeze(0)
    result = result.float()
    return result


def reshape_then_acc_adj_sort(event_tensor, augment=None, **kwargs):
    # Accumulate events to create a 2 * H * W image (DiST)

    # Augment data
    if augment is not None:
        event_tensor = augment(event_tensor)

    H = kwargs.get('height', IMAGE_H)
    W = kwargs.get('width', IMAGE_W)

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]

    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)
    pos_count = pos_count.float()

    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)
    neg_count = neg_count.float()

    # clip count
    pos_unique_count = torch.unique(pos_count, return_counts=True)[1]
    pos_sum_subset = torch.cumsum(pos_unique_count, dim=0)
    pos_th_clip = pos_sum_subset[pos_sum_subset < H * W * CLIP_COUNT_RATE].shape[0]
    pos_count[pos_count > pos_th_clip] = pos_th_clip

    neg_unique_count = torch.unique(neg_count, return_counts=True)[1]
    neg_sum_subset = torch.cumsum(neg_unique_count, dim=0)
    neg_th_clip = neg_sum_subset[neg_sum_subset < H * W * CLIP_COUNT_RATE].shape[0]
    neg_count[neg_count > neg_th_clip] = neg_th_clip

    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length

    # Get pos, neg time
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_min_out, _ = scatter_min(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W).float()
    pos_min_out = pos_min_out.reshape(H, W).float()
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_min_out, _ = scatter_min(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W).float()
    neg_min_out = neg_min_out.reshape(H, W).float()

    pos_min_out[pos_count == 0] = 1.0
    neg_min_out[neg_count == 0] = 1.0

    # Get temporal discount
    pos_disc = torch.zeros_like(pos_count)
    neg_disc = torch.zeros_like(neg_count)

    patch_size = 5

    pos_neighbor_count = patch_size ** 2 * torch.nn.functional.avg_pool2d(pos_count.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)
    neg_neighbor_count = patch_size ** 2 * torch.nn.functional.avg_pool2d(neg_count.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)

    pos_disc = (torch.nn.functional.max_pool2d(pos_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2) + 
        torch.nn.functional.max_pool2d(-pos_min_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)) / \
        (pos_neighbor_count)
    neg_disc = (torch.nn.functional.max_pool2d(neg_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2) + 
        torch.nn.functional.max_pool2d(-neg_min_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)) / \
        (neg_neighbor_count)

    pos_out[pos_count > 0] = (pos_out[pos_count > 0] - DISC_ALPHA * pos_disc.squeeze()[pos_count > 0])
    pos_out[pos_out < 0] = 0
    pos_out[pos_neighbor_count.squeeze() == 1.0] = 0
    neg_out[neg_count > 0] = (neg_out[neg_count > 0] - DISC_ALPHA * neg_disc.squeeze()[neg_count > 0])
    neg_out[neg_out < 0] = 0
    neg_out[neg_neighbor_count.squeeze() == 1.0] = 0

    pos_out = pos_out.reshape(H * W)
    neg_out = neg_out.reshape(H * W)

    pos_val, pos_idx = torch.sort(pos_out)
    neg_val, neg_idx = torch.sort(neg_out)
    
    pos_unq, pos_cnt = torch.unique_consecutive(pos_val, return_counts=True)
    neg_unq, neg_cnt = torch.unique_consecutive(neg_val, return_counts=True)

    pos_sort = torch.zeros_like(pos_out)
    neg_sort = torch.zeros_like(neg_out)

    pos_sort[pos_idx] = torch.repeat_interleave(torch.arange(pos_unq.shape[0]), pos_cnt).float() / pos_unq.shape[0]
    neg_sort[neg_idx] = torch.repeat_interleave(torch.arange(neg_unq.shape[0]), neg_cnt).float() / neg_unq.shape[0]

    pos_sort = pos_sort.reshape(H, W)
    neg_sort = neg_sort.reshape(H, W)

    result = torch.stack([pos_sort, neg_sort], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()

    return result


# Augmentation Modules


def random_shift_events(event_tensor, max_shift=20, resolution=(224, 224)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
    event_tensor[:, 0] += x_shift
    event_tensor[:, 1] += y_shift

    valid_events = (event_tensor[:, 0] >= 0) & (event_tensor[:, 0] < W) & (event_tensor[:, 1] >= 0) & (event_tensor[:, 1] < H)
    event_tensor = event_tensor[valid_events]

    return event_tensor


def random_flip_events_along_x(event_tensor, resolution=(224, 224), p=0.5):
    H, W = resolution

    if np.random.random() < p:
        event_tensor[:, 0] = W - 1 - event_tensor[:, 0]

    return event_tensor


def random_time_flip(event_tensor, resolution=(224, 224), p=0.5):
    if np.random.random() < p:
        event_tensor = torch.flip(event_tensor, [0])
        event_tensor[:, 2] = event_tensor[0, 2] - event_tensor[:, 2]
        event_tensor[:, 3] = - event_tensor[:, 3]  # Inversion in time means inversion in polarity
    return event_tensor


def random_time_select(event_tensor, resolution=(224, 224), slice_length=20000):
    slice_start = min(np.ceil(len(event_tensor) * np.random.random()), len(event_tensor) - slice_length)

    event_tensor = event_tensor[slice_start: slice_start + slice_length]
    return event_tensor


def random_trajectory_shift(event_tensor, resolution=(224, 224)):
    theta = np.random.random() * 2 * np.pi
    scale = np.random.random()
    speed_x = MAX_TRAJECTORY_SPEED * np.cos(theta) * scale
    speed_y = MAX_TRAJECTORY_SPEED * np.sin(theta) * scale

    time_tensor = event_tensor[:, 2] - event_tensor[0, 2]  # Time is given in seconds

    event_tensor[:, 0] += time_tensor * speed_x
    event_tensor[:, 1] += time_tensor * speed_y

    H, W = resolution
    valid_events = (event_tensor[:, 0] >= 0) & (event_tensor[:, 0] < W) & (event_tensor[:, 1] >= 0) & (event_tensor[:, 1] < H)
    event_tensor = event_tensor[valid_events]

    return event_tensor


def random_jump(event_tensor, resolution=(224, 224)):
    sample_prob = torch.zeros_like(event_tensor[:, 0]).fill_(np.random.random() * 0.7 + 0.3)
    event_tensor = event_tensor[torch.bernoulli(sample_prob).long().nonzero(as_tuple=True)]

    return event_tensor


def random_noise_inject(event_image, cfg):
    H = event_image.shape[1]
    W = event_image.shape[2]
    C = event_image.shape[0]

    change_prob = 0.5
    change_tgt = torch.bernoulli(torch.zeros(C).fill_(change_prob)).bool()

    tgt_event_image = event_image[change_tgt]

    if tgt_event_image.shape[0] != 0:
        # Hot pixels
        hot_level = abs(torch.randn(1).item() * 0.001)
        num_hot = int(torch.rand(1).item() * hot_level * H * W)
        pixel_location = torch.cat([torch.randint(0, H, size=(num_hot, 1)), torch.randint(0, W, size=(num_hot, 1))], dim=-1)
        
        tgt_event_image = tgt_event_image.permute(1, 2, 0)
        tgt_event_image[(pixel_location[:, 1], pixel_location[:, 0])] = torch.rand(num_hot, tgt_event_image.shape[-1]) * 0.2 + 0.8
        tgt_event_image = tgt_event_image.permute(2, 0, 1)

        # Background activity
        back_level = abs(torch.randn(1).item() * 0.08)
        num_back = int(torch.rand(1).item() * back_level * H * W)
        pixel_location = torch.cat([torch.randint(0, H, size=(num_back, 1)), torch.randint(0, W, size=(num_back, 1))], dim=-1)
        
        tgt_event_image = tgt_event_image.permute(1, 2, 0)
        tgt_event_image[(pixel_location[:, 1], pixel_location[:, 0])] = torch.rand(num_back, tgt_event_image.shape[-1]) * 0.8
        tgt_event_image = tgt_event_image.permute(2, 0, 1)

    event_image[change_tgt] = tgt_event_image
    return event_image


def base_augment(mode):
    assert mode in ['train', 'eval']

    if mode == 'train':
        def augment(event):
            event = random_time_flip(event, resolution=(IMAGE_H, IMAGE_W))
            event = random_flip_events_along_x(event)
            event = random_shift_events(event)
            return event
        return augment

    elif mode == 'eval':
        return None


def robust_augment(mode):
    assert mode in ['train', 'eval']

    if mode == 'train':
        def augment(event):
            event = random_time_flip(event, resolution=(IMAGE_H, IMAGE_W))
            event = random_flip_events_along_x(event)
            event = random_trajectory_shift(event, resolution=(IMAGE_H, IMAGE_W))
            event = random_shift_events(event)

            return event
        return augment

    elif mode == 'eval':
        return None


# Event modification functions
def spatial_flip(event):
    flip_augment_event = event.clone().detach()
    flip_augment_event[:, 0] = IMAGE_W - 1 - event[:, 0]

    return flip_augment_event


def temporal_flip(event):
    flip_augment_event = torch.flip(event, [0])
    flip_augment_event[:, 2] = flip_augment_event[0, 2] - flip_augment_event[:, 2]
    flip_augment_event[:, 3] = - flip_augment_event[:, 3]  # Inversion in time means inversion in polarity

    return flip_augment_event


def polarity_flip(event):
    flip_augment_event = event.clone().detach()
    flip_augment_event[:, 3] = - event[:, 3]

    return flip_augment_event


# Representation modification functions
def mutual_remove(event_img, src_chn, tgt_chn, window_size:int = 13):
    if len(event_img.shape) == 3:
        # Remove noise in channel specified by tgt_chn using information from tgt_chn
        mask = (torch.nn.functional.avg_pool2d(event_img[src_chn: src_chn + 1].unsqueeze(1), kernel_size=window_size, stride=1, padding=window_size // 2) == 0)
        mask = mask.squeeze()
        event_img[tgt_chn, mask] = 0.
    else:
        # Remove noise in channel specified by tgt_chn using information from tgt_chn
        mask = (torch.nn.functional.avg_pool2d(event_img[:, src_chn: src_chn + 1], kernel_size=window_size, stride=1, padding=window_size // 2) == 0)
        if tgt_chn == 0:
            mask = torch.cat([mask, torch.zeros_like(mask)], dim=1)
        else:
            mask = torch.cat([torch.zeros_like(mask), mask], dim=1)
        event_img[mask] = 0.

    return event_img

def neighbor_mask(event_img, window_size:int = 13):
    # Return mask whose values are 1 for regions with a window_size neighborhood with zero values
    mask = (torch.nn.functional.avg_pool2d(event_img.unsqueeze(0), kernel_size=window_size, stride=1, padding=window_size // 2) == 0)
    mask = mask.reshape(event_img.shape[0], event_img.shape[1], event_img.shape[2])

    return mask

def count_remove(event_img, orig_event, threshold):
    # Remove events accoring to event count
    count_img = reshape_then_acc_count_pol(orig_event)
    count_img = count_img / count_img.max(-1, keepdim=True).values

    event_img[count_img < threshold] = 0.
    return event_img


def conditional_remove(event_img, score_func):
    # Remove parts of event_img according to values from score_func
    pos_count = (event_img[0] != 0).sum()
    neg_count = (event_img[1] != 0).sum()

    rat = pos_count / neg_count
    if score_func(rat).item() < 0.1:
        event_img = mutual_remove(event_img, 0, 1)
    elif score_func(rat).item() > 0.9:
        event_img = mutual_remove(event_img, 1, 0)
    return event_img


def score_fusion(event_img, score_func):
    # Remove parts of event_img according to values from score_func
    pos_count = (event_img[0] != 0).sum()
    neg_count = (event_img[1] != 0).sum()

    rat = pos_count / neg_count
    score = score_func(rat).item()

    pos_guide_img = mutual_remove(event_img, 0, 1)
    neg_guide_img = mutual_remove(event_img, 1, 0)
    
    if score < 0.5:
        event_img = 2 * score * event_img + (1. - 2 * score) * pos_guide_img
    else:
        score = 1 - score
        event_img = 2 * score * event_img + (1. - 2 * score) * neg_guide_img
    
    return event_img

class ImageNetDataset(Dataset):
    def __init__(self, cfg, mode='train'):
        super(ImageNetDataset, self).__init__()
        self.mode = mode
        self.train_file = open(cfg.train_file, 'r').readlines()
        self.val_file = open(cfg.val_file, 'r').readlines()

        self.train_file = [(Path(s.strip())) for s in self.train_file]
        self.val_file = [(Path(s.strip())) for s in self.val_file]

        if mode == 'train':
            self.map_file = self.train_file
        elif mode == 'val':
            self.map_file = self.val_file
        elif mode == 'test':
            self.map_file = self.val_file

        self.labels = [s.split()[1].strip() for s in open(cfg.label_map, 'r').readlines()]
        self.labels = sorted(self.labels[:1000])

        if getattr(cfg, 'trim_class_count', None) is not None:
            self.labels = self.labels[:cfg.trim_class_count]
            self.map_file = list(filter(lambda s: s.parent.stem in self.labels, self.map_file))

        self.label_map = {s: idx for idx, s in enumerate(self.labels)}

        self.cfg = cfg
        self.augment_type = getattr(cfg, 'augment_type', None)
        self.loader_type = getattr(cfg, 'loader_type', None)
        self.parser_type = getattr(cfg, 'parser_type', 'normal')
        assert self.parser_type in ['normal', 'tree_search']

        # Choose parser (event path -> (N, 4) event tensor)
        if self.parser_type == 'normal':
            self.event_parser = self.augment_parser(parse_event)
        elif self.parser_type == 'tree_search':
            self.event_parser = self.augment_parser(tree_parse_event)
        
        # Choose how to augment samples for use in UDA training
        self.uda_augment_slice = getattr(self.cfg, 'uda_augment_slice', False)
        self.uda_multi_slice = getattr(self.cfg, 'uda_multi_slice', True)  # Defaults to using multiple slice

        # Choose loader ((N, 4) event tensor -> Network input)
        if self.loader_type == 'reshape_then_acc_time':
            load_func = reshape_then_acc_time
        elif self.loader_type == 'reshape_then_acc_count':
            load_func = reshape_then_acc_count
        elif self.loader_type == 'reshape_then_acc_diff_dist':
            load_func = reshape_then_acc_diff_dist
        elif self.loader_type == 'reshape_then_flat_pol':
            load_func = reshape_then_flat_pol
        elif self.loader_type == 'reshape_then_flat':
            load_func = reshape_then_flat
        elif self.loader_type == 'reshape_then_acc_time_pol':
            load_func = reshape_then_acc_time_pol
        elif self.loader_type == 'reshape_then_acc_count_pol':
            load_func = reshape_then_acc_count_pol
        elif self.loader_type == 'reshape_then_acc_exp':
            load_func = reshape_then_acc_exp
        elif self.loader_type == 'reshape_then_acc_sort':
            load_func = reshape_then_acc_sort
        elif self.loader_type == 'reshape_then_acc_intensity':
            load_func = reshape_then_acc_intensity
        elif self.loader_type == 'reshape_then_acc_adj_sort':
            load_func = reshape_then_acc_adj_sort
        elif self.loader_type == 'composable':
            loader_list = self.cfg.loader_list
            from real_cnn_model.data.loader_utils import ComposableLoader
            load_func = ComposableLoader(loader_list)
        elif self.loader_type == 'mixed':
            loader_list = self.cfg.loader_list
            from real_cnn_model.data.loader_utils import MixedLoader
            load_func = MixedLoader(loader_list)
        else:
            raise ValueError("Invalid Loader Type!")
        
        self.loader = lambda x, augment: load_func(x, augment)

        # Choose whether to use only one channel
        self.mutual_remove_channel = getattr(self.cfg, 'mutual_remove_channel', None)
        self.z_test_channel = getattr(self.cfg, 'z_test_channel', False)
        if self.z_test_channel:
            # Load original data statistics
            pos_count = torch.load('data_stats/v0_full_pos.pt')
            neg_count = torch.load('data_stats/v0_full_neg.pt')

            std_p = pos_count.std()
            std_n = neg_count.std()

            mu_p = pos_count.mean()
            mu_n = neg_count.mean()

            vp = pos_count - mu_p
            vn = neg_count - mu_n

            rho = torch.sum(vp * vn) / (torch.sqrt(torch.sum(vp ** 2)) * torch.sqrt(torch.sum(vn ** 2)))
            
            # Apply Geary transform
            self.geary_trans = lambda rat: (mu_n * rat - mu_p) / torch.sqrt(std_n ** 2 * rat ** 2 - 2 * rho * std_p * std_n * rat + std_p ** 2)

            # CDF score
            self.gauss_score = lambda rat: 1 / 2 * (1 + torch.erf(self.geary_trans(rat) / 2 ** (1 / 2)))
            self.z_score = lambda x, n: n ** (1 / 2) * x
            self.p_value = lambda x: 1 / 2 * (1 + torch.erf(x / 2 ** (1 / 2)))

    def augment_parser(self, parser):
        def new_parser(event_path):
            return parser(event_path, self.cfg)
        return new_parser

    def __getitem__(self, idx):
        event_path = self.map_file[idx]
        label = self.label_map[event_path.parent.stem]

        # Load and optionally reshape event from event_path
        orig_event = self.event_parser(event_path)
        augment_mode = 'train' if self.mode == 'train' else 'eval'
        if self.augment_type == 'base_augment':
            event = self.loader(orig_event, augment=base_augment(augment_mode))
        else:
            event = self.loader(orig_event, augment=None)
        
        if self.mode == 'train':
            if self.uda_multi_slice:
                slice_augment_list = [self.event_parser(event_path) for _ in range(getattr(self.cfg, 'num_sentry_augment', 3))]
            else:
                slice_augment_list = [orig_event for _ in range(getattr(self.cfg, 'num_sentry_augment', 3))]
            
            if self.uda_augment_slice:
                aug_list = random.choices([temporal_flip, spatial_flip, polarity_flip, lambda x: x], 
                    k=getattr(self.cfg, 'num_sentry_augment', 3))
                slice_augment_event = [self.loader(aug(slice_augment_event), augment=None) for aug, slice_augment_event in 
                    zip(aug_list, slice_augment_list)]
            else:
                slice_augment_event = [self.loader(slice_augment_event, augment=None) for slice_augment_event in slice_augment_list]

            slice_augment_event = torch.stack(slice_augment_event, dim=0)  # (B, C, H, W)

            return event, label, slice_augment_event

        else:
            return event, label

    def __len__(self):
        return len(self.map_file)

