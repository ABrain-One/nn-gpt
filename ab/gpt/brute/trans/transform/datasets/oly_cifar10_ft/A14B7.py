import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=2, fill=(132, 74, 64), padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(p=0.89),
        transforms.RandomVerticalFlip(p=0.28),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])