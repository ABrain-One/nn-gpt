import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(98, 51, 243), padding_mode='edge'),
    transforms.RandomAffine(degrees=25, translate=(0.01, 0.07), scale=(1.05, 1.53), shear=(1.17, 8.09)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
