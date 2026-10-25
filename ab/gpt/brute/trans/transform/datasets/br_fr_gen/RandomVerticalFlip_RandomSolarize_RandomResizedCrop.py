import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.11),
    transforms.RandomSolarize(threshold=144, p=0.54),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.96), ratio=(0.91, 2.13)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
