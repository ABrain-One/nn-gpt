<td colspan=2>
<pre style="white-space: pre; font-family: &quot;Andale Mono&quot;;">
import torch<br />
import torchvision.transforms as transforms<br />
<br />
def transform(norm):<br />
\   return transforms.Compose([<br />
\   transforms.Pad(padding=4, fill=(42, 103, 19), padding_mode='constant'),<br />
\   transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.0), shear=(-5, 5)),<br />
\   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br />
\   transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0)),<br />
\   transforms.Resize((64, 64)),<br />
\   transforms.ToTensor(),<br />
\   transforms.Normalize(**norm)<br />
   ])
</pre>
</td>