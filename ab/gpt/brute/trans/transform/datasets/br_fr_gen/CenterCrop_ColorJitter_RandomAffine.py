import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.ColorJitter(brightness=1.0, contrast=1.06, saturation=1.09, hue=0.02),
    transforms.RandomAffine(degrees=13, translate=(0.15, 0.09), scale=(0.89, 1.81), shear=(2.9, 8.3)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
