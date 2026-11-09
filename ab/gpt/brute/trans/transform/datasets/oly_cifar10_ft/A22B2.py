import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(105, 195, 105), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.64),
    transforms.RandomResize(scale=(0.8, 0.93)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])