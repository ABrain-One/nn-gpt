import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(17, 13, 233), padding_mode='edge'),
    transforms.RandomAutocontrast(p=0.78),
    transforms.RandomAffine(degrees=14, translate=(0.08, 0.09), scale=(1.03, 1.58), shear=(1.95, 8.06)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
