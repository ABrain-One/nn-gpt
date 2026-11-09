import torch<br/>
import torchvision.transforms as transforms<br/>
&nbsp;<br/>
def transform(norm):<br/>
&nbsp;&nbsp;return transforms.Compose([<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomGrayscale(p=0.6),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomPosterize(bits=5, p=0.6),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(0.8, 1.2)),<br/>
&nbsp;&nbsp;&nbsp;transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),<br/>
&nbsp;&nbsp;&nbsp;transforms.RandomApply([transforms.GaussianBlur(3)], p=0.4),<br/>
&nbsp;&nbsp;&nbsp;transforms.Resize((64, 64)),<br/>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br/>
&nbsp;&nbsp;&nbsp;transforms.Normalize(*norm)<br/>
&nbsp;&nbsp;])<br/>