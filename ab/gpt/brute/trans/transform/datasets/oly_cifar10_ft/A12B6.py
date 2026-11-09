import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.83),
    transforms.Pad(padding=2, fill=(176, 191, 162), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.18, p=0.43),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])