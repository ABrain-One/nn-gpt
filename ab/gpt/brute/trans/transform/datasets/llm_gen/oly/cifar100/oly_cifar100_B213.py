import torch<br/>
import torchvision.transforms as transforms<br/><br/>
def transform(norm):<br/>
&nbsp; &nbsp; return transforms.Compose([<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(p=0.6),<br/>
&nbsp; &nbsp; &nbsp; transforms.ColorJitter(brightness=0.9, contrast=0.85, saturation=0.9, hue=0.05),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomInvert(p=0.25),<br/>
&nbsp; &nbsp; &nbsp; transforms.Pad(padding=4, fill=(254, 171, 47)),<br/>
&nbsp; &nbsp; &nbsp; transforms.Resize((64,64)),<br/>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br/>
&nbsp; &nbsp; &nbsp; transforms.Normalize(*norm)<br/>
&nbsp; &nbsp; ])