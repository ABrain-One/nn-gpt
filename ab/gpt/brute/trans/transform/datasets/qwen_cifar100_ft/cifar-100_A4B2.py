import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=28), transforms.RandomGrayscale(p=0.86), transforms.RandomSolarize(threshold=134, p=0.13), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])