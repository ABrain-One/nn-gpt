import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(231, 103, 189), padding_mode='symmetric'),
    transforms.ColorJitter(brightness=0.85, contrast=1.12, saturation=1.0, hue=0.0),
    transforms.RandomEqualize(p=0.89),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
