import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.14),
    transforms.RandomAffine(degrees=27, translate=(0.05, 0.1), scale=(0.83, 1.59), shear=(0.3, 6.49)),
    transforms.Pad(padding=3, fill=(76, 15, 183), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
