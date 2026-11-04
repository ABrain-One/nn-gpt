import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(105, 173, 105), padding_mode='symmetric'),
    transforms.RandomVerticalFlip(p=0.85),
    transforms.RandomAdjustSharpness(sharpness_factor=1.31, p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])