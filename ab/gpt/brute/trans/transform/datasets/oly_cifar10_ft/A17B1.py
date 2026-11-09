import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(3, 108, 107), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.97, p=0.52),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])