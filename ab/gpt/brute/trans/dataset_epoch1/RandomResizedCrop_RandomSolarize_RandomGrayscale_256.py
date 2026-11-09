import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.93), ratio=(0.89, 1.33)),
    transforms.RandomSolarize(threshold=130, p=0.19),
    transforms.RandomGrayscale(p=0.72),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
