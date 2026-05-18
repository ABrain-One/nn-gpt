import re

para_dict = {'accuracy': 0.9, 'duration': 0.1, 'model_name': 'test'}
text = "Result: <<accuracy / duration>> ns⁻¹"
print("Before:", text)
text = text.format(**para_dict)
print("After format:", text)

text2 = "Result: {{accuracy / duration}} ns⁻¹"
print("Before 2:", text2)
text2 = text2.format(**para_dict)
print("After format 2:", text2)

