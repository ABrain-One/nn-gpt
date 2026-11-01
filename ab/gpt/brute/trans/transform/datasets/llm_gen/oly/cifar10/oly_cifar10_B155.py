import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(128, 128, 128), padding_mode='symmetric'), transforms.RandomSolarize(threshold=20, p=0.5), transforms.RandomAffine(degrees=5, translate=(0.2, 0.2), scale=(1.2, 1.3), shear=(-5, 8)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])