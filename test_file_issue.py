import os
import sys
from pathlib import Path

# Add repo root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from ab.nn.util.Train import train_new
import ab.nn.util.Const as Const

print("Original Const.out:", Const.out)

import ab.nn.util.Train as train_runtime
print("Original Train.out:", train_runtime.out)

isolated_out = f"out_nneval_tmp_pid_{os.getpid()}_TEST"
train_runtime.out = isolated_out
print("Set Train.out to:", train_runtime.out)

nn_code = '''
import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, in_shape, out_shape, prm, device):
        super().__init__()
        self.device = device
    def train_setup(self, prm):
        pass
    def learn(self, train_data):
        pass
'''

try:
    train_new(nn_code, 'img-classification', 'cifar-10', 'acc', {'lr': 0.01, 'batch': 10, 'dropout': 0.2, 'momentum': 0.9, 'transform': 'norm_256_flip', 'epoch': 1}, save_to_db=False)
except Exception as e:
    print(f"Exception caught: {type(e).__name__}: {e}")

print("Checking if isolated out directory exists:")
isolated_root = Path(Const.ab_root_path) / isolated_out
print(isolated_root, isolated_root.exists())
if isolated_root.exists():
    for root, dirs, files in os.walk(isolated_root):
        print("DIR:", root, dirs, files)
