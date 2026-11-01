import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.58), transforms.RandomPosterize(bits=5, p=0.15), transforms.RandomSolarize(threshold=104, p=0.88), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([0.499, 0.4358, 0.4811], [0.2292, 0.2253, 0.2281])])