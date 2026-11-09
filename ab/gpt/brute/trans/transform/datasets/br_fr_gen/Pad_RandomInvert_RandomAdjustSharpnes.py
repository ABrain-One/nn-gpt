import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(124, 28, 72), padding_mode='reflect'),
    transforms.RandomInvert(p=0.81),
    transforms.RandomAdjustSharpness(sharpness_factor=0.51, p=0.64),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
