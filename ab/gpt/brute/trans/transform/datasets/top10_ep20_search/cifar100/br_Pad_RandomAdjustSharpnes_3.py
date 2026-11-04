import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(136, 186, 245), padding_mode='constant'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
