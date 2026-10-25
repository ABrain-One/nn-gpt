import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(198, 169, 45), padding_mode='constant'), transforms.RandomAffine(degrees=10, translate=(0.0, 0.1), scale=(1.14, 1.69), shear=(4.04, 5.82)), transforms.RandomGrayscale(p=0.26), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])