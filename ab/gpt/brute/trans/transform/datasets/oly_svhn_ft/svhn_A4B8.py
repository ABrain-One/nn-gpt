import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(160, 104, 22), padding_mode='reflect'), transforms.RandomEqualize(p=0.57), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.499, 0.474, 0.411], std=[0.262, 0.2591, 0.2584])])