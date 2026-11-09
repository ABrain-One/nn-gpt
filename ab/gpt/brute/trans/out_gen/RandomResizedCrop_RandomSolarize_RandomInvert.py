import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.91), ratio=(1.0, 2.08)),
    transforms.RandomSolarize(threshold=5, p=0.82),
    transforms.RandomInvert(p=0.73),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
