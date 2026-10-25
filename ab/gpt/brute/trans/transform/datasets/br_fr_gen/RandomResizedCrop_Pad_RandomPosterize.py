import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.92), ratio=(0.88, 1.83)),
    transforms.Pad(padding=1, fill=(133, 117, 232), padding_mode='reflect'),
    transforms.RandomPosterize(bits=5, p=0.69),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
