import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.4), transforms.RandomEqualize(p=0.33), transforms.RandomPosterize(bits=4, p=0.73), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])