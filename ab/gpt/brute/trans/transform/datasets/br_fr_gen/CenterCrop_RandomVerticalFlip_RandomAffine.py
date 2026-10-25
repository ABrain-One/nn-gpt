import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomVerticalFlip(p=0.24),
    transforms.RandomAffine(degrees=17, translate=(0.05, 0.03), scale=(1.14, 1.73), shear=(0.35, 6.55)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
