import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.71, 0.96), ratio=(1.11, 2.82)), transforms.RandomEqualize(p=0.77), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.2827, 0.2829, 0.2807], std=[0.2819, 0.2821, 0.2812])])