import torch<br/>
import torchvision.transforms as transforms<br/>
&nbsp;<br/>
def transform(norm):<br/>
&nbsp; &nbsp; return transforms.Compose([<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomVerticalFlip(p=0.75),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomGrayscale(p=0.5),<br/>
&nbsp; &nbsp; &nbsp; transforms.Resize(80),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomCrop(64),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomPerspective(distortion_scale=0.3, p=0.5),<br/>
&nbsp; &nbsp; &nbsp; transforms.ColorJitter(brightness=0.8, contrast=1.0, saturation=1.0, hue=0.05),<br/>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br/>
&nbsp; &nbsp; &nbsp; transforms.Normalize(*norm)<br/>
&nbsp; &nbsp; ])