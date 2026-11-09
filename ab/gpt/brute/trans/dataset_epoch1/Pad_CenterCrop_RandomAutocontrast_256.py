import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(217, 4, 28), padding_mode='symmetric'),
    transforms.CenterCrop(size=32),
    transforms.RandomAutocontrast(p=0.53),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
