import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=1, fill=(19, 169, 149), padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(p=0.83),
        transforms.RandomAutocontrast(p=0.74),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])