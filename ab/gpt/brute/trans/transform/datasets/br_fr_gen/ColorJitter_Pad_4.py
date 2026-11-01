import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.97, contrast=0.99, saturation=1.15, hue=0.06),
    transforms.Pad(padding=2, fill=(91, 239, 57), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
