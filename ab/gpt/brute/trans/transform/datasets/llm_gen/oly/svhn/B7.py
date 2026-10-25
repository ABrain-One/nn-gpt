import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=29), transforms.RandomPerspective(distortion_scale=0.29, p=0.67), transforms.RandomSolarize(threshold=161, p=0.24), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])