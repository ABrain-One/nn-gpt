import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=95, p=0.89),
    transforms.RandomPosterize(bits=7, p=0.42),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
