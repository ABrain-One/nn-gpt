import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.49),
    transforms.RandomPerspective(distortion_scale=0.26, p=0.22),
    transforms.RandomEqualize(p=0.58),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
