import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.96), ratio=(1.12, 2.2)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.44, p=0.48),
    transforms.Pad(padding=1, fill=(152, 254, 52), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
