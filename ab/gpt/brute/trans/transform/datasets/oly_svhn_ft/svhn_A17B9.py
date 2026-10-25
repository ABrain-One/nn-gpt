import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(191, 19, 19), padding_mode='symmetric'), transforms.RandomSolarize(threshold=32, p=0.74), transforms.RandomResizedCrop(size=32, scale=(0.72, 0.97), ratio=(1.06, 2.24)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])