import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(147, 156, 174), padding_mode='edge'),
    transforms.RandomCrop(size=29),
    transforms.RandomEqualize(p=0.36),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
