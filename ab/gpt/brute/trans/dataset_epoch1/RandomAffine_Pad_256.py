import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=2, translate=(0.08, 0.17), scale=(1.04, 1.57), shear=(4.56, 9.56)),
    transforms.Pad(padding=0, fill=(190, 8, 139), padding_mode='symmetric'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
