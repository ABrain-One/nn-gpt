import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.93), ratio=(1.22, 2.21)),
    transforms.RandomHorizontalFlip(p=0.57),
    transforms.RandomSolarize(threshold=170, p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
