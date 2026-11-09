import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.89), ratio=(1.05, 1.46)),
    transforms.RandomGrayscale(p=0.18),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.49),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
