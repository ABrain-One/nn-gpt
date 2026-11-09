import torch<br/>
from torchvision import transforms<br/>
&nbsp;<br/>
def transform(norm):<br/>
 &nbsp;return transforms.Compose([<br/>
 &nbsp;&nbsp;transforms.Pad(padding=4, fill=(0.4914, 0.4822, 0.4465), padding_mode='constant'),<br/>
 &nbsp;&nbsp;transforms.RandomResizedCrop(40, scale=(0.8, 1.0), ratio=(0.9, 1.1)),<br/>
 &nbsp;&nbsp;transforms.RandomRotation(degrees=(-10, 10)),<br/>
 &nbsp;&nbsp;transforms.RandomHorizontalFlip(),<br/>
 &nbsp;&nbsp;transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),<br/>
 &nbsp;&nbsp;transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.5),<br/>
 &nbsp;&nbsp;transforms.Resize((64, 64)),<br/>
 &nbsp;&nbsp;transforms.ToTensor(),<br/>
 &nbsp;&nbsp;transforms.Normalize(*norm)<br/>
 &nbsp;])<br/>