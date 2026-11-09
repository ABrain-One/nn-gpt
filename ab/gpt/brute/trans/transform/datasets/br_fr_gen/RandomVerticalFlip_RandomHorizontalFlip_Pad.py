import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.33),
    transforms.RandomHorizontalFlip(p=0.73),
    transforms.Pad(padding=3, fill=(233, 245, 171), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
