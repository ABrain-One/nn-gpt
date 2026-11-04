import torch
import torchvision.transforms as transforms
import random

class StochasticGaussianBlur(transforms.GaussianBlur):

    def __init__(self, p=0.5, kernel_size=3, sigma_min=0.51, sigma_max=1.91):
        super().__init__(kernel_size=kernel_size, sigma=sigma_max)
        self.p = p
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def forward(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            return super().forward(img, sigma)
        return img

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(38, 66, 30), padding_mode='constant'), transforms.RandomVerticalFlip(p=0.86), StochasticGaussianBlur(p=0.5, sigma_min=0.51, sigma_max=1.91), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])