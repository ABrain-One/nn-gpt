import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.29),
    transforms.RandomAffine(degrees=25, translate=(0.14, 0.03), scale=(1.13, 1.48), shear=(1.36, 9.94)),
    transforms.Pad(padding=1, fill=(212, 141, 188), padding_mode='reflect'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
