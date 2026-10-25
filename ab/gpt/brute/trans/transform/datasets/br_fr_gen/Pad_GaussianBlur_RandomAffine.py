import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(71, 42, 83), padding_mode='constant'),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.64, 1.48)),
    transforms.RandomAffine(degrees=5, translate=(0.07, 0.17), scale=(0.87, 1.37), shear=(3.56, 6.02)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
