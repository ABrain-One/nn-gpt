<pre><code>&lt;tr&gt;import torch&lt;/tr&gt;
&lt;tr&gt;import torchvision.transforms as transforms&lt;/tr&gt;
&lt;tr&gt;
def transform(norm):&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; return transforms.Compose([&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; transforms.RandomResizedCrop(size=32, scale=&amp;#40;0.8, 1.0&amp;#41;, ratio=&amp;#40;0.9, 1.1&amp;#41;, interpolation=transforms.InterpolationMode.BICUBIC),&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; transforms.RandomVerticalFlip(p=0.2),&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; transforms.RandomHorizontalFlip(p=0.5),&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; transforms.RandomApply([&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; &amp;nbsp; transforms.GaussianBlur(kernel_size=3, sigma=1.0),&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; ], p=0.2),&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; transforms.RandomInvert(p=0.56),&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; transforms.Resize(&amp;#40;64, 64&amp;#41;),&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; transforms.ToTensor(),&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; &amp;nbsp; transforms.Normalize(*norm)&lt;/tr&gt;
&lt;tr&gt;&amp;nbsp; &amp;nbsp; ])&lt;/tr&gt;
</code></pre>