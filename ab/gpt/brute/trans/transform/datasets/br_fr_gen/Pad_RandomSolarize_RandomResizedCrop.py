import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(251, 238, 38), padding_mode='edge'),
    transforms.RandomSolarize(threshold=163, p=0.13),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.96), ratio=(0.99, 2.18)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
