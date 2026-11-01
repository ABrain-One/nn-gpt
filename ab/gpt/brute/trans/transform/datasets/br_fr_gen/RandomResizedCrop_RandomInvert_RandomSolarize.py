import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.84), ratio=(1.06, 2.63)),
    transforms.RandomInvert(p=0.43),
    transforms.RandomSolarize(threshold=161, p=0.89),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
