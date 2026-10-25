import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomAffine(degrees=14, translate=(0.19, 0.05), scale=(1.0, 1.99), shear=(2.24, 5.84)),
    transforms.Pad(padding=2, fill=(142, 70, 187), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
