import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(72, 143, 255), padding_mode='reflect'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.38, p=0.5),
    transforms.RandomRotation(degrees=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
