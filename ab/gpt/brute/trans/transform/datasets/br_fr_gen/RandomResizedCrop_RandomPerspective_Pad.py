import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.8), ratio=(1.31, 1.99)),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.8),
    transforms.Pad(padding=1, fill=(237, 60, 252), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
