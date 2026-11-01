import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(252, 17, 142), padding_mode='constant'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.75),
    transforms.RandomCrop(size=26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])