import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(37, 245, 126), padding_mode='symmetric'),
    transforms.RandomInvert(p=0.14),
    transforms.RandomAffine(degrees=5, translate=(0.07, 0.02), scale=(1.08, 1.3), shear=(3.83, 7.71)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
