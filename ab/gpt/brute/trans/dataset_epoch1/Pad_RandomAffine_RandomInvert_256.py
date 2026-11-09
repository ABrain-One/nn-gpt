import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(20, 13, 204), padding_mode='symmetric'),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.11), scale=(0.8, 1.66), shear=(3.77, 9.4)),
    transforms.RandomInvert(p=0.73),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
