import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.78), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.31, 1.55)), transforms.RandomSolarize(threshold=151, p=0.45), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=norm[0], std=norm[1])])