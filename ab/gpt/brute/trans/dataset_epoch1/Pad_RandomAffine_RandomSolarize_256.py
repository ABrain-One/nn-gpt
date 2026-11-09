import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(101, 69, 242), padding_mode='edge'),
    transforms.RandomAffine(degrees=15, translate=(0.18, 0.06), scale=(1.19, 1.7), shear=(3.21, 5.66)),
    transforms.RandomSolarize(threshold=146, p=0.48),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
