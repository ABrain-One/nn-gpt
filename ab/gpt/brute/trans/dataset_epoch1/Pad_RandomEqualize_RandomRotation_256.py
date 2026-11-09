import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(118, 131, 46), padding_mode='edge'),
    transforms.RandomEqualize(p=0.77),
    transforms.RandomRotation(degrees=14),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
