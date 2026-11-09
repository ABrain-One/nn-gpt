import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.96), ratio=(1.06, 2.51)),
    transforms.ColorJitter(brightness=0.84, contrast=1.1, saturation=0.85, hue=0.07),
    transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.39),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
