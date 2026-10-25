import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(105, 173, 137), padding_mode='edge'),
    transforms.RandomSolarize(threshold=103, p=0.54),
    transforms.RandomAdjustSharpness(sharpness_factor=1.28, p=0.75),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])