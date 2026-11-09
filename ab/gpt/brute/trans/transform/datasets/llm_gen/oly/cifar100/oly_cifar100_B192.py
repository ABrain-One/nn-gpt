import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>
&nbsp;&nbsp;return transforms.Compose([<br/>
&nbsp;&nbsp;&nbsp;transforms.Pad(padding=4, fill=(186, 58, 81), padding_mode='constant'),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-5.0, 5.0)),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomVerticalFlip(p=0.5),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomCrop(size=32),<br/>
&nbsp;&nbsp;&nbsp;transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
&nbsp;&nbsp;&nbsp;transforms.Resize((64, 64)),<br/>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br/>
&nbsp;&nbsp;&nbsp;transforms.Normalize(*norm)<br/>
&nbsp;&nbsp;])<br/>