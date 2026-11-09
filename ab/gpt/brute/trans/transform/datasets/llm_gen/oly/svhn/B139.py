import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(111, 227, 54), padding_mode='edge'), transforms.RandomRotation(21), transforms.RandomAffine(degrees=2, translate=(0.09, 0.11), scale=(0.89, 1.27), shear=(1.75, 6.18)), transforms.RandomHorizontalFlip(p=0.5), transforms.Grayscale(num_output_channels=3), transforms.RandomAdjustSharpness(1.63, p=0.52), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4391, 0.4452, 0.4785], std=[0.2952, 0.2952, 0.302])])