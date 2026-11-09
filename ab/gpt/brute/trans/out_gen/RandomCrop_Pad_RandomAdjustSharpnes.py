import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.Pad(padding=5, fill=(77, 116, 215), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.62, p=0.44),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
