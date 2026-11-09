import random
import os
import re
import torchvision.transforms as transforms
import torch
import itertools
import argparse

# available transforms with parameter generators
variable_transforms = [
    {
        'name': 'CenterCrop',
        'params': lambda: {'size': random.randint(24, 32)}  # Reduced max size to fit CIFAR-10 (32x32)
    },
    {
        'name': 'Pad',
        'params': lambda: {
            'padding': random.randint(0, 5),  
            'fill': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            'padding_mode': random.choice(["'constant'", "'edge'", "'reflect'", "'symmetric'"])
        }
    },
    {
        'name': 'RandomCrop',
        'params': lambda: {'size': random.randint(24, 32)}  
    },
    {
        'name': 'RandomHorizontalFlip',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    },
    {
        'name': 'RandomVerticalFlip',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    },
    {
        'name': 'RandomResizedCrop',
        'params': lambda: {
            'size': 32, # Fixed size for CIFAR-10
            'scale': (round(random.uniform(0.5, 0.8), 2), round(random.uniform(0.8, 1.0), 2)),
            'ratio': (round(random.uniform(0.75, 1.33), 2), round(random.uniform(1.33, 3.0), 2)) 
        }
    },
    {
        'name': 'ColorJitter',
        'params': lambda: {
            'brightness': round(random.uniform(0.8, 1.2), 2),
            'contrast': round(random.uniform(0.8, 1.2), 2),
            'saturation': round(random.uniform(0.8, 1.2), 2),
            'hue': round(random.uniform(0.0, 0.1), 2)  
        }
    },
    {
        'name': 'RandomRotation',
        'params': lambda: {'degrees': random.randint(0, 30)}
    },
    {
        'name': 'RandomAffine',
        'params': lambda: {
            'degrees': random.randint(0, 30),
            'translate': (round(random.uniform(0.0, 0.2), 2), round(random.uniform(0.0, 0.2), 2)),
            'scale': (round(random.uniform(0.8, 1.2), 2), round(random.uniform(1.2, 2.0), 2)),
            'shear': (round(random.uniform(0, 5), 2), round(random.uniform(5, 10), 2))  
        }
    },
    {
    'name': 'Grayscale',
    'params': lambda: {'num_output_channels': 3}  # Always output 3 channels
    },
    {
    'name': 'RandomGrayscale',  
    'params': lambda: {
        'p': round(random.uniform(0.1, 0.9), 2)
    }
    },
    {
        'name': 'RandomPerspective',
        'params': lambda: {
            'distortion_scale': round(random.uniform(0.1, 0.3), 2),  # Reduced distortion
            'p': round(random.uniform(0.1, 0.9), 2)
        }
    },
    {
        'name': 'GaussianBlur',
        'params': lambda: {
            'kernel_size': random.choice([3, 5]),
            'sigma': (round(random.uniform(0.1, 1.0), 2), round(random.uniform(1.0, 2.0), 2)) 
        }
    },
    {
        'name': 'RandomInvert',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    },
    {
        'name': 'RandomPosterize',
        'params': lambda: {
            'bits': random.choice([4, 5, 6, 7, 8]),
            'p': round(random.uniform(0.1, 0.9), 2)
        }
    },
    {
        'name': 'RandomSolarize',
        'params': lambda: {
            'threshold': random.randint(0, 255),
            'p': round(random.uniform(0.1, 0.9), 2)
        }
    },
    {
        'name': 'RandomAdjustSharpness',
        'params': lambda: {
            'sharpness_factor': round(random.uniform(0.5, 2.0), 2),
            'p': round(random.uniform(0.1, 0.9), 2)
        }
    },
    {
        'name': 'RandomAutocontrast',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    },
    {
        'name': 'RandomEqualize',
        'params': lambda: {'p': round(random.uniform(0.1, 0.9), 2)}
    }
]


# Default settings
OUTPUT_DIR = "ab/gpt/brute/trans/out"
MAX_FILENAME_LENGTH = 250

filename_counter = {}

def generate_transform_file(transforms_list, directory):
    """
    Generate a transform file with 1, 2 or 3 transforms + fixed transforms.
    """
    # Build base filename from transform names
    name_parts = [t['name'][:20] for t in transforms_list]
    base_name = '_'.join(name_parts)
    base_name = re.sub(r'[^a-zA-Z0-9_]', '', base_name)

    # Handle name clashes
    count = filename_counter.get(base_name, 0) + 1
    filename_counter[base_name] = count
    output_filename = f"{base_name}_{count}.py" if count > 1 else f"{base_name}.py"


    # Generate transform code lines
    transform_lines = []
    
    for i, vt in enumerate(transforms_list):
        name = vt['name']
        params = vt['params']()
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        transform_lines.append(f"transforms.{name}({param_str})")
        
        
    # Fixed transforms
    fixed_transforms = [
        'transforms.Resize((64,64))',
        'transforms.ToTensor()',
        'transforms.Normalize(*norm)'
    ]

    # Combine with fixed transforms
    all_transforms = transform_lines + fixed_transforms
    transforms_code = ',\n    '.join(all_transforms)

    # Write to file
    full_path = os.path.join(directory, output_filename)
    with open(full_path, 'w') as f:
        f.write(f"""import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    {transforms_code}
])
""")

    print(f"Saved: {full_path}")

    return full_path



def generate_files(transform_no, file_num, output_dir):
    """
    Generate transform files with 1, 2 or 3 transforms + fixed transforms.

    Args:
        transform_no: Number of transforms
        file_num: Number of files to generate
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    if transform_no == 1:
        all_combinations = variable_transforms
    elif transform_no == 2:
        all_combinations = list(itertools.permutations(variable_transforms, 2))
    elif transform_no == 3:
        all_combinations = list(itertools.permutations(variable_transforms, 3))
    else:
        raise ValueError("transform_no must be 1, 2, or 3")

    # Create infinite cycle
    all_combinations_cycle = itertools.cycle(all_combinations)

     # Clear all files in the directory
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(output_dir, exist_ok=True)


    # Generate files
    for _ in range(file_num):
        transforms_list = next(all_combinations_cycle)
        if transform_no == 1:
            transforms_list = [transforms_list]
        generate_transform_file(transforms_list, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate transform files")
    parser.add_argument(
        '-t', '--transform_no', type=int, default=3,
        help= "Number of transforms per file (1, 2, or 3)"
    )
    parser.add_argument(
        '-n', '--file_num', type=int, default=200,
        help=f"Number of files to generate"
    )
    parser.add_argument(
        '-o', '--output_dir', type=str, default=OUTPUT_DIR,
        help=f"Output directory"
    )

    args = parser.parse_args()

    generate_files(args.transform_no, args.file_num, args.output_dir)

if __name__ == "__main__":
    main()