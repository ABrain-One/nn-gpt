import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(62, 122, 81), padding_mode='reflect'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.58, p=0.39),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
