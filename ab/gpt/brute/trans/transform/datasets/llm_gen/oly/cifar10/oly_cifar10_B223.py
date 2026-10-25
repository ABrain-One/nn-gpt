import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(125, 123, 114), padding_mode='symmetric'), transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=10), transforms.RandomEqualize(p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])