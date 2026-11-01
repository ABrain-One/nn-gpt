import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.81),
    transforms.RandomAdjustSharpness(sharpness_factor=0.52, p=0.78),
    transforms.Pad(padding=4, fill=(166, 194, 189), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
