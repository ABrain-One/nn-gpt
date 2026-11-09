import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.8), ratio=(0.83, 2.79)),
    transforms.Pad(padding=2, fill=(78, 112, 144), padding_mode='symmetric'),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
