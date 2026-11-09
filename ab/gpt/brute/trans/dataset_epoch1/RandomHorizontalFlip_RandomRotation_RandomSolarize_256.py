import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.42),
    transforms.RandomRotation(degrees=27),
    transforms.RandomSolarize(threshold=210, p=0.73),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
