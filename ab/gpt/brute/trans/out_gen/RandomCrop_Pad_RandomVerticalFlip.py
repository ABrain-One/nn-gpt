import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.Pad(padding=4, fill=(102, 161, 244), padding_mode='symmetric'),
    transforms.RandomVerticalFlip(p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
