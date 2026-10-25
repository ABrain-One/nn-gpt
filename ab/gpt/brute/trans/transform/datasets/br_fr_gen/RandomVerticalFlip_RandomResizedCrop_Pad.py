import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.46),
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.89), ratio=(1.02, 1.48)),
    transforms.Pad(padding=4, fill=(35, 205, 30), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
