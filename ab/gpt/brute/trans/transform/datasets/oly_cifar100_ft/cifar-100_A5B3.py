import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.77, 0.81), ratio=(1.32, 1.68)), transforms.RandomEqualize(p=0.65), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.52, 0.53, 0.58], std=[0.22, 0.23, 0.27])])