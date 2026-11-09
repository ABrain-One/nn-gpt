import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.Pad(padding=1, fill=(165, 11, 54), padding_mode='edge'),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.82), ratio=(1.05, 1.34)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
