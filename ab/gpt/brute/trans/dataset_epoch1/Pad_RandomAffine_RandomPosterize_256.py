import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(160, 95, 183), padding_mode='constant'),
    transforms.RandomAffine(degrees=18, translate=(0.13, 0.03), scale=(1.03, 1.77), shear=(4.08, 5.7)),
    transforms.RandomPosterize(bits=7, p=0.22),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
