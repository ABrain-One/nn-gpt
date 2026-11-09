import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.03, contrast=1.19, saturation=0.89, hue=0.04),
    transforms.Pad(padding=3, fill=(233, 207, 143), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
