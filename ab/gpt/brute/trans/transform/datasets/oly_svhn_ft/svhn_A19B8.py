import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=27), transforms.RandomPerspective(distortion_scale=0.14, p=0.14), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.22, 0.221, 0.2242], std=[0.105, 0.102, 0.102])])