import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(234, 122, 150), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.97, p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])