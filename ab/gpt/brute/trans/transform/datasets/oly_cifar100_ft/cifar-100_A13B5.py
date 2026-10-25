import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=28), transforms.RandomPerspective(distortion_scale=0.21, p=0.89), transforms.Pad(padding=1, fill=(118, 4, 189), padding_mode='edge'), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])