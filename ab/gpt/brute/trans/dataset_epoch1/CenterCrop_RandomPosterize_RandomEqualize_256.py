import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.RandomPosterize(bits=4, p=0.71),
    transforms.RandomEqualize(p=0.62),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
