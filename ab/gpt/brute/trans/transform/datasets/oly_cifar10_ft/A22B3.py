import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(244, 7, 205), padding_mode='constant'),
    transforms.RandomPosterize(bits=4, p=0.34),
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.81), ratio=(0.78, 2.9)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])