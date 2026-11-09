import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.83), ratio=(0.96, 2.63)),
    transforms.RandomHorizontalFlip(p=0.47),
    transforms.CenterCrop(size=27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
