import sys
import tqdm
import numpy as np
from pathlib import Path
import random
import torch
from scipy.optimize import minimize
import matplotlib.pyplot as plt

file_list = sys.argv[1]

SENSOR_H = 480
SENSOR_W = 640
IMAGE_H = 224
IMAGE_W = 224
VISUALIZE = True
LENGTH = 50000
START_IDX = 0
OBJECTIVE = 'gradient'

def load_event(event_path):
    # Returns time-shifted numpy array event from event_path
    event = np.load(event_path)['event_data']

    event = np.vstack([event['x'], event['y'], event['t'], event['p'].astype(np.uint8)]).T
    event = event.astype(np.float)

    # Account for non-zero minimum time
    if event[:, 2].min() != 0:
        event[:, 2] -= event[:, 2].min()

    # Account for int-type timestamp
    # event[:, 2] /= 1000000

    # Account for zero polarity
    if event[:, 3].min() >= -0.5:
        event[:, 3][event[:, 3] <= 0.5] = -1

    event[:, 0] *= (IMAGE_W / SENSOR_W)
    event[:, 1] *= (IMAGE_H / SENSOR_H)

    return event


def display_event(event):
    event_image = np.zeros([IMAGE_H, IMAGE_W])
    coords = event[:, :2].astype(np.int32)
    event_image[(coords[:, 1], coords[:, 0])] = 1.0

    plt.imshow(event_image)
    plt.show()

def warp_event(event_path):
    event = load_event(event_path)
    speed = np.zeros(2)

    display_event(event)

    def tgt_func(x):
        tgt_event = np.array(event[START_IDX:START_IDX + LENGTH])

        tgt_event[:, 0] = tgt_event[:, 0] + x[0] * (tgt_event[START_IDX, 2] - tgt_event[:, 2])
        tgt_event[:, 1] = tgt_event[:, 1] + x[1] * (tgt_event[START_IDX, 2] - tgt_event[:, 2])
        coords = tgt_event[:, :2].astype(np.int32)
        coords[:, 0] = np.clip(coords[:, 0], 0, IMAGE_W - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, IMAGE_H - 1)

        event_image = np.zeros([IMAGE_H, IMAGE_W])
        event_image[(coords[:, 1], coords[:, 0])] = 1.0

        plt.imshow(event_image)
        plt.show()

        obj_value = 0.0
        if OBJECTIVE == 'proj_cnt':
            obj_value = np.average(event_image)
        
        elif OBJECTIVE == 'gradient':
            gy, gx = np.gradient(event_image)
            gnorm = np.sqrt(gx**2 + gy**2)
            obj_value = -np.average(gnorm)
        
        elif OBJECTIVE == 'variance':
            obj_value = -np.var(event_image)
        
        print(obj_value)
        
        return obj_value

    result = minimize(tgt_func, speed, bounds=[(-1.0 / 1000, 1.0 / 1000), (-1.0 / 1000, 1.0 / 1000)])
    speed = result.x

    event[:, 0] = event[:, 0] + speed[0] * (event[START_IDX, 2] - event[:, 2])
    event[:, 1] = event[:, 1] + speed[1] * (event[START_IDX, 2] - event[:, 2])

    event[:, 0] = np.clip(event[:, 0], 0, IMAGE_W - 1)
    event[:, 1] = np.clip(event[:, 1], 0, IMAGE_H - 1)
    display_event(event)
    
    import pdb; pdb.set_trace()
    return result


def save_event(event_tensor, save_path):
    pass

if __name__ == '__main__':
    file_list = open(file_list, 'r').readlines()
    file_list = [Path(s.strip()) for s in file_list]

    for event_path in tqdm.tqdm(file_list):
        result_event = warp_event(event_path)
        save_path = 'tmp'
        save_event(result_event, save_path)
    