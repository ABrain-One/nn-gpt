import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(198, 101, 45), padding_mode='reflect'),
    transforms.ColorJitter(brightness=1.16, contrast=0.91, saturation=1.0, hue=0.07),
    transforms.RandomAffine(degrees=4, translate=(0.02, 0.12), scale=(1.2, 1.93), shear=(2.84, 5.9)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
