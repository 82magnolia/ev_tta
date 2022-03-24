import sys
import tqdm
import numpy as np
from pathlib import Path
import random
import torch
import pickle

file_list = sys.argv[1]
label_list = sys.argv[2]
file_list = open(file_list, 'r').readlines()
file_list = [s.strip() for s in file_list]

labels = [s.split()[1].strip() for s in open(label_list, 'r').readlines()]
labels = sorted(labels[:1000])
label_map = {s: idx for idx, s in enumerate(labels)}
label_data = {s:[] for s in labels}


SENSOR_H = 224
SENSOR_W = 224
IMAGE_H = 224
IMAGE_W = 224

# stat_list = ['pos_count', 'neg_count', 'x_mean', 'y_mean', 'x_range', 'y_range', 'speed']
stat_list = ['speed']

def load_event(event_path):
    # Returns time-shifted numpy array event from event_path
    event = torch.load(event_path)
    return event


def get_speed(event):
    if (event[-1, 2] - event[0, 2]) * 1000000. < 0.1:
        return 0.

    return len(event) / ((event[-1, 2] - event[0, 2]) * 1000000.)


def get_bundle_speed(event_path):
    event = load_event(event_path)
    event = event[:20000]

    return len(event) / ((event[-1, 2] - event[0, 2]) * 1000000.)


def get_stats(event_path):
    event = load_event(event_path)
    
    label_name = Path(event_path).parent.name

    if len(event) == 0:
        value_list = [0.0 for _ in stat_list]
        label_data[label_name].append(value_list)
        return value_list

    coords = event[:, :2].long()

    event_image = torch.zeros([SENSOR_H, SENSOR_W])
    event_image[(coords[:, 1], coords[:, 0])] = 1.0

    value_list = []

    if 'pos_count' in stat_list:
        pos_count = len(event[event[:, -1] > 0])
        value_list.append(pos_count)

    if 'neg_count' in stat_list:
        neg_count = len(event[event[:, -1] < 0])
        value_list.append(neg_count)

    if 'x_mean' in stat_list:
        value_list.append(event[:, 0].mean())

    if 'y_mean' in stat_list:
        value_list.append(event[:, 1].mean())

    if 'x_range' in stat_list:
        value_list.append(event[:, 0].max() - event[:, 0].min())

    if 'y_range' in stat_list:
        value_list.append(event[:, 1].max() - event[:, 1].min())

    if 'speed' in stat_list:
        value_list.append(get_speed(event))
    
    label_data[label_name].append(value_list)

    return value_list

result_list = [get_stats(event_path) for event_path in tqdm.tqdm(file_list)]
result_tensor = torch.tensor(result_list)
torch.save(result_tensor, Path(sys.argv[1]).parent / (Path(sys.argv[1]).stem + "_total_data.pt"))

for label_name in tqdm.tqdm(label_data.keys()):
    label_data[label_name] = torch.tensor(label_data[label_name])

torch.save(label_data, (Path(sys.argv[1]).parent / (Path(sys.argv[1]).stem + "_label_data.pt")))

