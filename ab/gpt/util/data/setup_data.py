# setup_data.py

import os
import subprocess
from ab.gpt.conf.Conf import GITHUB_REPO_DIR, DATASET_DESC_DIR

def ensure_repo_cloned(repo_url, local_name, branch="main"):
    """
    Clones the specified GitHub repository into the local directory if it does not exist.
    """
    repo_path = os.path.join(GITHUB_REPO_DIR, local_name)
    if not os.path.isdir(repo_path):
        print(f"Cloning {repo_url} (branch: {branch}) into {repo_path}...")
        os.makedirs(GITHUB_REPO_DIR, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--single-branch", "--branch", branch, repo_url, repo_path],
            check=True
        )
    else:
        print(f"Repository '{local_name}' already exists at {repo_path}.")

def ensure_dataset_file(file_name, content=None):
    """
    Creates a dataset description file if it does not exist.
    """
    file_path = os.path.join(DATASET_DESC_DIR, file_name)
    if not os.path.isfile(file_path):
        print(f"Creating dataset file: {file_path}")
        os.makedirs(DATASET_DESC_DIR, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content or "No content specified.")
    else:
        print(f"Dataset file '{file_name}' already exists at {file_path}.")

def run_setup():
    """
    Clones key GitHub repositories that contain technical, code-rich content
    and creates dataset description files for multiple datasets.
    """
    ensure_repo_cloned("https://github.com/osmr/pytorchcv.git", "pytorchcv", branch="master")
    ensure_repo_cloned("https://github.com/pytorch/vision.git", "pytorch_vision", branch="main")
    ensure_repo_cloned("https://github.com/fastai/fastai.git", "fastai", branch="master")

    ensure_dataset_file(
        file_name="cifar10.txt",
        content=(
            "The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, "
            "with 6,000 images per class. Often used for image classification tasks."
        )
    )
    ensure_dataset_file(
        file_name="mnist.txt",
        content=(
            "The MNIST dataset has 70,000 28x28 grayscale images of handwritten digits, "
            "divided into 60,000 training and 10,000 testing images. Widely used for digit recognition."
        )
    )
    ensure_dataset_file(
        file_name="cifar100.txt",
        content=(
            "The CIFAR-100 dataset has 60,000 32x32 color images in 100 classes, "
            "with 600 images per class. It is a more challenging version of CIFAR-10."
        )
    )
    print("Data setup complete.")

if __name__ == "__main__":
    run_setup()
