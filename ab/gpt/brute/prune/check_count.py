import sys
from pathlib import Path
import os

# Dynamically add workspace roots to path based on script location
script_path = Path(__file__).resolve()
gw_root = script_path.parent.parent.parent.parent.parent  # Points to .../nn-gpt
ds_root = gw_root.parent / 'nn-dataset'          # Points to .../nn-dataset

sys.path.append(str(ds_root))
sys.path.append(str(gw_root))
sys.path.append(str(gw_root / 'ab' / 'gpt' / 'util'))

from ab.nn.util.Const import core_nn_cls
import Const as const

print(f"Models in Const.py: {len(core_nn_cls)}")
out_dir = const.synth_dir(const.epoch_dir('0'))
folders = [f for f in os.listdir(out_dir) if f.startswith('B')]
print(f"Output Folders (B*): {len(folders)}")

# Find missing index
present_indices = sorted([int(f[1:]) for f in folders])
all_indices = list(range(len(core_nn_cls)))
missing = set(all_indices) - set(present_indices)
print(f"Missing Indices: {missing}")
if missing:
    for idx in missing:
         print(f"Missing Model: {core_nn_cls[idx]}")
