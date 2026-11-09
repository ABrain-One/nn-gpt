import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.Pad(padding=2, fill=(184, 4, 59), padding_mode='constant'),
    transforms.RandomAffine(degrees=13, translate=(0.06, 0.05), scale=(0.81, 1.51), shear=(4.33, 9.55)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
