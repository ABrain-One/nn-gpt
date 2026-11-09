import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(160, 191, 103), padding_mode='edge'), transforms.RandomSolarize(threshold=141, p=0.62), transforms.RandomEqualize(p=0.62), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.1541, 0.2022, 0.1814], std=[0.2729, 0.2678, 0.2642])])