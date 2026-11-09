import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(175, 203, 45), padding_mode='symmetric'), transforms.RandomEqualize(p=0.25), transforms.RandomAutocontrast(p=0.73), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])