import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(97, 173, 131), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.66, p=0.62),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.95), ratio=(1.19, 1.59)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
