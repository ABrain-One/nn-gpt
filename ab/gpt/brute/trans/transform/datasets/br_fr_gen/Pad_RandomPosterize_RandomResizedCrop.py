import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(105, 206, 189), padding_mode='edge'),
    transforms.RandomPosterize(bits=7, p=0.51),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.96), ratio=(1.06, 1.6)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
