import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(223, 20, 235), padding_mode='symmetric'),
    transforms.RandomSolarize(threshold=26, p=0.36),
    transforms.RandomAffine(degrees=2, translate=(0.19, 0.17), scale=(1.15, 1.45), shear=(1.47, 6.23)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
