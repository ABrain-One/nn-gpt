import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(57, 21, 143), padding_mode='edge'),
    transforms.RandomAffine(degrees=25, translate=(0.06, 0.04), scale=(1.14, 1.44), shear=(4.05, 8.22)),
    transforms.CenterCrop(size=26),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
