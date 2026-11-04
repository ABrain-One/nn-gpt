import torch
import torchvision.transforms as transforms
import PIL.Image as Image

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8 * 32, 1.25 * 32), interpolation=Image.BICUBIC), transforms.RandomAffine(degrees=5, translate=(0.05, 0.1), scale=(0.9, 1.1), shear=(-3, 3)), transforms.RandomEqualize(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])