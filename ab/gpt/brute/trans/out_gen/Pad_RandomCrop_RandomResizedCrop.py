import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(93, 36, 253), padding_mode='reflect'),
    transforms.RandomCrop(size=31),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.83), ratio=(1.27, 2.77)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
