import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(209, 170, 203), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.72, p=0.79),
    transforms.RandomVerticalFlip(p=0.62),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
