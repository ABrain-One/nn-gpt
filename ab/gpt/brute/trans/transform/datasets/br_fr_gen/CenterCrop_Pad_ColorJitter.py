import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.Pad(padding=1, fill=(89, 230, 156), padding_mode='symmetric'),
    transforms.ColorJitter(brightness=0.8, contrast=0.96, saturation=1.14, hue=0.05),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
