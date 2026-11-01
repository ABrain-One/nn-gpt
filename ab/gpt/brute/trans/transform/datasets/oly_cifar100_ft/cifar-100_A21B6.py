import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.66, 0.95), ratio=(0.83, 1.83)), transforms.RandomSolarize(threshold=242, p=0.16), transforms.RandomInvert(p=0.78), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.481, 0.461, 0.782), (0.281, 0.289, 0.711))])