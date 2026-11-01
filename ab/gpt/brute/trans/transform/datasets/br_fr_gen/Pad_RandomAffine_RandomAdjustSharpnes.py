import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(166, 178, 47), padding_mode='symmetric'),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.13), scale=(0.99, 1.81), shear=(4.42, 6.69)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.55, p=0.22),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
