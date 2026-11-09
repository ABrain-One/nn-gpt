import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.9), ratio=(1.11, 2.86)),
    transforms.Pad(padding=1, fill=(235, 173, 65), padding_mode='symmetric'),
    transforms.RandomEqualize(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
