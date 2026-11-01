import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.97), ratio=(0.88, 2.67)),
    transforms.Pad(padding=3, fill=(230, 152, 97), padding_mode='constant'),
    transforms.CenterCrop(size=32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
