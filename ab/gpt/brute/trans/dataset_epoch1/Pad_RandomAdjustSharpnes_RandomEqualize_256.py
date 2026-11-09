import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(204, 104, 114), padding_mode='constant'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.37, p=0.63),
    transforms.RandomEqualize(p=0.88),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
