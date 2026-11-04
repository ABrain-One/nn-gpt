import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.34),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.38),
    transforms.RandomSolarize(threshold=172, p=0.56),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
