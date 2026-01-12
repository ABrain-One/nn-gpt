import Const as const
import os

print(f"Models in Const.py: {len(const.core_nns)}")
out_dir = const.synth_dir(const.epoch_dir('0'))
folders = [f for f in os.listdir(out_dir) if f.startswith('B')]
print(f"Output Folders (B*): {len(folders)}")

# Find missing index
present_indices = sorted([int(f[1:]) for f in folders])
all_indices = list(range(len(const.core_nns)))
missing = set(all_indices) - set(present_indices)
print(f"Missing Indices: {missing}")
if missing:
    for idx in missing:
         print(f"Missing Model: {const.core_nns[idx]}")
