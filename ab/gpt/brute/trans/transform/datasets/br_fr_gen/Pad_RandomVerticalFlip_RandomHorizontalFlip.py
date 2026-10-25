import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(61, 198, 215), padding_mode='reflect'),
    transforms.RandomVerticalFlip(p=0.27),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
