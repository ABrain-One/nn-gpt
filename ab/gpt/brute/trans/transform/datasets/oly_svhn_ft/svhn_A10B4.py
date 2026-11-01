import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 0.94), ratio=(1.02, 2.8)), transforms.RandomInvert(p=0.28), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.269, 0.23, 0.2423], [0.2874, 0.2821, 0.281])])