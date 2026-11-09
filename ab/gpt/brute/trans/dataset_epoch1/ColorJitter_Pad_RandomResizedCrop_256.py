import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.12, contrast=0.93, saturation=0.81, hue=0.07),
    transforms.Pad(padding=3, fill=(81, 72, 142), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.96), ratio=(0.88, 1.51)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
