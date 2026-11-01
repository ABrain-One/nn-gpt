import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.85),
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.95), ratio=(0.94, 1.68)),
    transforms.RandomSolarize(threshold=192, p=0.62),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
