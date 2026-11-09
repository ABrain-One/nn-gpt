import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(5, 176, 109), padding_mode='symmetric'),
    transforms.RandomRotation(degrees=7),
    transforms.RandomCrop(size=29),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
