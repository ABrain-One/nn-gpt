import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(191, 198, 64), padding_mode='constant'),
    transforms.RandomEqualize(p=0.28),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.97), ratio=(1.09, 2.98)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
