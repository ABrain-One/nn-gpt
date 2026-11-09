import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(193, 242, 29), padding_mode='edge'),
    transforms.RandomAffine(degrees=11, translate=(0.19, 0.05), scale=(0.93, 1.65), shear=(3.17, 7.65)),
    transforms.ColorJitter(brightness=0.86, contrast=1.06, saturation=0.87, hue=0.02),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
