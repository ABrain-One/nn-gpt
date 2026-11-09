import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.Pad(padding=3, fill=(177, 244, 177), padding_mode='symmetric'),
    transforms.RandomAffine(degrees=8, translate=(0.1, 0.14), scale=(1.11, 1.6), shear=(4.04, 9.4)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
