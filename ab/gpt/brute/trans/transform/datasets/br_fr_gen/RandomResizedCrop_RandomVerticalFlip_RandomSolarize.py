import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.86), ratio=(0.89, 1.44)),
    transforms.RandomVerticalFlip(p=0.71),
    transforms.RandomSolarize(threshold=144, p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
