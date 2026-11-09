import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(189, 245, 239), padding_mode='edge'),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.96), ratio=(1.09, 1.92)),
    transforms.RandomPosterize(bits=7, p=0.67),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
