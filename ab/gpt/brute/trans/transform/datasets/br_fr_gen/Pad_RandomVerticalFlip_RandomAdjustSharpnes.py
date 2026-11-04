import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(81, 177, 243), padding_mode='symmetric'),
    transforms.RandomVerticalFlip(p=0.58),
    transforms.RandomAdjustSharpness(sharpness_factor=1.85, p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
