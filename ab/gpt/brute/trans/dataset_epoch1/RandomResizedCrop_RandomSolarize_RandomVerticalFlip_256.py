import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.82), ratio=(0.87, 1.7)),
    transforms.RandomSolarize(threshold=221, p=0.16),
    transforms.RandomVerticalFlip(p=0.37),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
