import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(120, 31, 162), padding_mode='symmetric'), transforms.RandomAffine(degrees=16, translate=(0.06, 0.05), scale=(0.87, 1.68), shear=(4.04, 9.47)), transforms.RandomEqualize(p=0.16), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])