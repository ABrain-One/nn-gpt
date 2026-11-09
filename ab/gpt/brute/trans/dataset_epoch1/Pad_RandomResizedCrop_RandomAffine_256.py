import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(174, 146, 74), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.82), ratio=(0.88, 1.53)),
    transforms.RandomAffine(degrees=17, translate=(0.01, 0.15), scale=(0.93, 1.67), shear=(3.94, 5.41)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
