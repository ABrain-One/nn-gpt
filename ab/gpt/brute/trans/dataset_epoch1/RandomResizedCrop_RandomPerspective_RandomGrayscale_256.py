import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.94), ratio=(1.31, 2.03)),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.46),
    transforms.RandomGrayscale(p=0.3),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
