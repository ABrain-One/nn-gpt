import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(26), transforms.RandomGrayscale(p=0.2), transforms.RandomApply([transforms.Autocontrast()], p=0.2), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])