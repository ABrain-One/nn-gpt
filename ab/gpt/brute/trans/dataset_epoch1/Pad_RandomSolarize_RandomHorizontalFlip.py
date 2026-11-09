import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(57, 29, 30), padding_mode='reflect'),
    transforms.RandomSolarize(threshold=167, p=0.13),
    transforms.RandomHorizontalFlip(p=0.14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
