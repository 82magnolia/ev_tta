import sys
import tqdm
import numpy as np
from pathlib import Path
import random
import torch

file_list = sys.argv[1]
file_list = open(file_list, 'r').readlines()
idx_list = [int(s.strip().split(':')[1]) for s in file_list]
file_list = [(Path(s.strip().split(':')[0])) for s in file_list]


SENSOR_H = 480
SENSOR_W = 640
IMAGE_H = 224
IMAGE_W = 224

stat_list = ['speed', 'proj_cnt', 'var', 'grad']


def load_event(event_path):
    # Returns time-shifted numpy array event from event_path
    event = np.load(event_path)

    event = np.vstack([event['x_pos'], event['y_pos'], event['timestamp'], event['polarity']]).T

    # Account for non-zero minimum time
    if event[:, 2].min() != 0:
        event[:, 2] -= event[:, 2].min()

    event = event.astype(np.float)

    # Account for int-type timestamp
    event[:, 2] /= 1000000

    # Account for zero polarity
    if event[:, 3].min() >= -0.5:
        event[:, 3][event[:, 3] <= 0.5] = -1

    event[:, 0] *= (IMAGE_W / SENSOR_W)
    event[:, 1] *= (IMAGE_H / SENSOR_H)

    return event


def get_speed(event_path):
    event = load_event(event_path)

    return len(event) / ((event[-1, 2] - event[0, 2]) * 1000000.)


def get_bundle_speed(event_path):
    event = load_event(event_path)
    event = event[:20000]

    return len(event) / ((event[-1, 2] - event[0, 2]) * 1000000.)


def get_stats(event_path, idx=None):
    event = torch.from_numpy(load_event(event_path))
    length = 20000

    if idx is not None:
        coords = event[idx: idx + length, :2].long()
    else:
        coords = event[:, :2].long()

    event_image = torch.zeros([IMAGE_H, IMAGE_W])
    event_image[(coords[:, 1], coords[:, 0])] = 1.0

    value_list = []

    if 'speed' in stat_list:
        length = 20000

        avg_speed = 0.0
        num_samples = 10
        
        if idx is not None:
            event = event[idx: idx + length]
            avg_speed = len(event) / ((event[-1, 2] - event[0, 2]) * 1000000.)
        else:
            if len(event) > length:
                for _ in range(num_samples):
                    start = random.choice(range(len(event) - length + 1))
                    sample_event = event[start: start + length]
                    avg_speed += len(sample_event) / ((sample_event[-1, 2] - sample_event[0, 2]) * 1000000.)
                
                avg_speed = avg_speed / num_samples
            else:
                avg_speed += len(event) / ((event[-1, 2] - event[0, 2]) * 1000000.)
        
        value_list.append(avg_speed)
    
    if 'proj_cnt' in stat_list:
        proj_cnt = torch.mean(event_image).item()

        value_list.append(proj_cnt)

    if 'grad' in stat_list:
        gy, gx = np.gradient(event_image.numpy())
        gnorm = np.sqrt(gx**2 + gy**2)
        sharpness = np.average(gnorm)
        value_list.append(sharpness)
    
    if 'var' in stat_list:
        var = torch.var(event_image)
        value_list.append(var)

    return value_list


# print("Average speed: ", sum([get_speed(event_path) for event_path in tqdm.tqdm(file_list)]) / len(file_list))
# print("Average bundle speed: ", sum([get_bundle_speed(event_path) for event_path in tqdm.tqdm(file_list)]) / len(file_list))

#result_list = [get_stats(event_path) for event_path in tqdm.tqdm(file_list)]

result_list = [get_stats(event_path, idx) for event_path, idx in tqdm.tqdm(zip(file_list, idx_list), total=len(file_list))]
result_tensor = torch.tensor(result_list)
torch.save(result_tensor, Path(sys.argv[1]).parent / (Path(sys.argv[1]).stem + "_speed.pt"))
