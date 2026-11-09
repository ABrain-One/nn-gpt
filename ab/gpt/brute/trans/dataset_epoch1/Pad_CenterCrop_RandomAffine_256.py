import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(12, 85, 224), padding_mode='constant'),
    transforms.CenterCrop(size=26),
    transforms.RandomAffine(degrees=29, translate=(0.02, 0.1), scale=(0.91, 1.4), shear=(1.76, 7.57)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
