import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(123, 86, 94), padding_mode='symmetric'),
    transforms.ColorJitter(brightness=1.1, contrast=1.08, saturation=0.93, hue=0.1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
