import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.ColorJitter(brightness=0.93, contrast=0.95, saturation=1.04, hue=0.07),
    transforms.Pad(padding=5, fill=(155, 19, 151), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
