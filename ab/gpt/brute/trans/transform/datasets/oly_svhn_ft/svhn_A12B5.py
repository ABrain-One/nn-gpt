import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomInvert(p=0.48), transforms.RandomGrayscale(p=0.69), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])