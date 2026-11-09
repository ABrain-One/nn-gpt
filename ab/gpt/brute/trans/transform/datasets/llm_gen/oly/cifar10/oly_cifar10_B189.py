tags.</
You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are given a competitive programming problem. Please reason step by step about the solution, then provide a complete implementation in C++17 (with all dependencies) that is thoroughly optimized for both speed and memory usage.

Your solution must read input from standard input (cin), write output to standard output (cout).
Do not include any debug prints or additional output.

Put your final solution within a single code block:
```cpp
<your code here>
```

# Problem

Little Elephant loves machine learning very much! He decided to train neural networks for computer vision tasks.

For training he wants to use dataset, which he calls C*own faces. And he wants to release it under MIT license this winter. But first he needs to prepare it. Could you help him and write script for automatic faces detection and cropping?

The script should detect all faces on the image, crop them and stack together in one image. One face should occupy exactly 20% of total size and keep aspect ratio.

Automatic face detection is really hard so for simplicity we will use three rules instead:

1. Each image has exactly one or zero eyes.
2. Each face has at least one eye.
3. Smallest rectangle, that meets next two conditions, is a face:   contains at least one eye;  distance between eyes more than or equal to the eye width.

## Constraints
Time limit per test: 2.0 seconds
Memory limit per test: 256.0 megabytes

## Input Format
The first line contains two integers h and w (1 ≤ h, w ≤ 1000) — height and width of the image.

Each of the following h lines contains w characters between '.' and '*' inclusive — encoded image.

## Output Format
If there is no face in the image, print one word "NO".

Otherwise print coordinates of rectangle with rounded border, which contains detected faces.

Format: "CWHR" where C — width of face, W — height of face, H — height of face, R — rotation in degrees ($$R \in \{0, 90, 180, 270\}$$). All numbers are integers.

## Examples
```input
5 8
....*...
..*.*..
*******.
**...***
...*....
```
```output
NO
```
-----
```input
5 8
.*.*.....
*..*.....
*********
*.o*.*.*.
.........
```
```output
64 46 46 90
```

## Note
In the second sample there is an image of man with mustache. His face is 64 pixels in width and 46 pixels in height and it is rotated by 90 degrees. Size of the eyes is 1px * 1px ("o" in the image). The distance between eyes is 3 pixels.