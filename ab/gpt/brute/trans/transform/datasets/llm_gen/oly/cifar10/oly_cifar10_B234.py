import torch
import torchvision.transforms as transforms
import random

class GaussianBlur(object):

    def __init__(self, prob=0.5, kernel_size=(3, 3), sigma=(0.1, 2.0)):
        self.prob = prob
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, img):
        if random.random() < self.prob:
            return self.blur(img)
        return img

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(70, 175, 131), padding_mode='symmetric'), transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), GaussianBlur(prob=0.5), transforms.ToTensor(), transforms.Normalize(*norm)])