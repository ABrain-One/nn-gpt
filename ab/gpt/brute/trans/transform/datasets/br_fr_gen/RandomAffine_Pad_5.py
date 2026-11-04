import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=29, translate=(0.14, 0.03), scale=(0.82, 1.33), shear=(4.68, 9.78)),
    transforms.Pad(padding=4, fill=(238, 134, 192), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
