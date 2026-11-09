import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.83), ratio=(1.15, 2.48)),
    transforms.Pad(padding=4, fill=(34, 128, 17), padding_mode='symmetric'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
