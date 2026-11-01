import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(248, 251, 110), padding_mode='reflect'), transforms.GaussianBlur(kernel_size=3, sigma=(0.71, 1.55)), transforms.RandomAffine(degrees=18, translate=(0.02, 0.08), scale=(0.9, 1.54), shear=(2.23, 8.7)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])