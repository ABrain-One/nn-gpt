import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.72, 0.9), ratio=(1.12, 1.72)), transforms.Pad(padding=2, fill=(109, 234, 240), padding_mode='reflect'), transforms.RandomSolarize(threshold=195, p=0.11), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])