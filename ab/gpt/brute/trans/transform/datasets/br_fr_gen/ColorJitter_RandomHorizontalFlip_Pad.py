import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.08, contrast=1.03, saturation=1.08, hue=0.07),
    transforms.RandomHorizontalFlip(p=0.69),
    transforms.Pad(padding=4, fill=(216, 255, 95), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
