import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.89), ratio=(1.26, 1.53)),
    transforms.RandomEqualize(p=0.15),
    transforms.Pad(padding=1, fill=(215, 46, 100), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
