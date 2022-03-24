import h5py
import numpy as np
import sys
import pathlib
import scipy.io
from tqdm import tqdm

mat_name = sys.argv[1]
orig_name = sys.argv[2]
new_root = sys.argv[3]

new_root = pathlib.Path(new_root)
key_name = pathlib.Path(mat_name).stem

f = h5py.File(mat_name, 'r')

data = f[key_name]

orig = scipy.io.loadmat(orig_name)

if 'Train' in key_name:
    labels = orig['Train_label']
    global_idx = ([int(s) for s in key_name if s.isdigit()][0] - 1) * 10000
else:
    labels = orig['Test_label']
    global_idx = 0

for idx in tqdm(range(len(data[0]))):
    event_data = np.array(f[data[0, idx]]).T
    event_data = event_data[:, [1, 2, 0, 3]]
    label_data = labels[global_idx + idx]
    np.savez_compressed(str(new_root / f"{global_idx + idx}.npz"), event_data=event_data, label_data=label_data)

f.close()
