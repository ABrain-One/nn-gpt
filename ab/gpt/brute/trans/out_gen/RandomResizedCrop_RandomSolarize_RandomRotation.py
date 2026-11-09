import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.9), ratio=(1.03, 1.55)),
    transforms.RandomSolarize(threshold=57, p=0.18),
    transforms.RandomRotation(degrees=27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
