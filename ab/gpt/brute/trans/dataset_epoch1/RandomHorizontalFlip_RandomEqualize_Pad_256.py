import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.16),
    transforms.RandomEqualize(p=0.35),
    transforms.Pad(padding=5, fill=(164, 114, 28), padding_mode='reflect'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
