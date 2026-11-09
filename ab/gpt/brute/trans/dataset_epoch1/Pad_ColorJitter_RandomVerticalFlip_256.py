import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(27, 78, 64), padding_mode='symmetric'),
    transforms.ColorJitter(brightness=1.15, contrast=0.94, saturation=1.04, hue=0.02),
    transforms.RandomVerticalFlip(p=0.84),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
