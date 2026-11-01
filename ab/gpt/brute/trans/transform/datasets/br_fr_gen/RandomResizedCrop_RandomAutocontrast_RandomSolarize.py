import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 1.0), ratio=(0.77, 2.78)),
    transforms.RandomAutocontrast(p=0.3),
    transforms.RandomSolarize(threshold=222, p=0.24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
