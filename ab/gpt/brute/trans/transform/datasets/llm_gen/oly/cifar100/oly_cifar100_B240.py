import torch
import torchvision.transforms as transforms

def transform(norm):
    transform1 = transforms.Compose([transforms.Pad(2, (121, 83, 208), 'symmetric'), transforms.Grayscale(3), transforms.RandomInvert(p=0.5)])
    transform2 = transforms.Compose([transforms.ColorJitter(0.87, 0.87, 0.99, 0.01), transforms.RandomPosterize(5, p=0.5)])
    rand_apply = transforms.RandomApply([transform1, transform2], p=0.5)
    return transforms.Compose([rand_apply, transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])