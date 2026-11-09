import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(213, 246, 184), padding_mode='symmetric'),
    transforms.RandomAffine(degrees=19, translate=(0.0, 0.16), scale=(1.15, 1.9), shear=(1.52, 8.29)),
    transforms.RandomVerticalFlip(p=0.83),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
