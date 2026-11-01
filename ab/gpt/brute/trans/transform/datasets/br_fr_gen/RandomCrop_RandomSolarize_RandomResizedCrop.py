import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.RandomSolarize(threshold=51, p=0.59),
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.81), ratio=(0.92, 1.99)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
