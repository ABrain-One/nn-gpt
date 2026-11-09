import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(2, 112, 4), padding_mode='reflect'),
    transforms.RandomVerticalFlip(p=0.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48], std=[0.21], inplace=True)
])