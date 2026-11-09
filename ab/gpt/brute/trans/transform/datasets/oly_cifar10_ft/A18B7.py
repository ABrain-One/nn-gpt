import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(251, 11, 39), padding_mode='reflect'),
    transforms.RandomAffine(degrees=3, translate=(0.07, 0.13), scale=(1.09, 1.69), shear=(4.33, 6.63)),
    transforms.RandomVerticalFlip(p=0.13),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])