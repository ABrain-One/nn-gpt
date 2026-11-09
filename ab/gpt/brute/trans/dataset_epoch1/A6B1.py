import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.Pad(padding=3, fill=(195, 173, 43), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.08, p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])