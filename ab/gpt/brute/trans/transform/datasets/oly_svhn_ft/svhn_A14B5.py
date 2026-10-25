import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=28), transforms.RandomPerspective(distortion_scale=0.3, p=0.33), transforms.RandomEqualize(p=0.67), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])