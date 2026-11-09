import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(250, 39, 48), padding_mode='edge'),
    transforms.RandomEqualize(p=0.16),
    transforms.CenterCrop(size=31),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
