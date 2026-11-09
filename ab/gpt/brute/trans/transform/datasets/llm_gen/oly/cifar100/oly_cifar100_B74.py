import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp; &nbsp; return transforms.Compose([<br>
&nbsp; &nbsp; &nbsp; transforms.Pad(padding=5, fill=(212, 9, 177), padding_mode='reflect'),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 2.45)),<br>
&nbsp; &nbsp; &nbsp; transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))], p=0.5),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomRotation(degrees=(-15, 15)),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(p=0.5),<br>
&nbsp; &nbsp; &nbsp; transforms.Resize(64),<br>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br>
&nbsp; &nbsp; &nbsp; transforms.Normalize(**norm)<br>
&nbsp; &nbsp; ])