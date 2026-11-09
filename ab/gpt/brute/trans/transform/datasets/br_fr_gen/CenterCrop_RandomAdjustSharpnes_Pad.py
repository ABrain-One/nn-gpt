import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.RandomAdjustSharpness(sharpness_factor=1.83, p=0.66),
    transforms.Pad(padding=1, fill=(215, 68, 57), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
