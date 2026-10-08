# data_loader.py

import os

def detect_libraries(text):
    """
    Detects key library mentions in the text (e.g., torch, tensorflow) and returns a list.
    """
    libs = []
    lower_text = text.lower()
    if "torch" in lower_text or "pytorch" in lower_text:
        libs.append("pytorch")
    if "tensorflow" in lower_text:
        libs.append("tensorflow")
    return libs

def chunk_code(text, min_chunk_length=50):
    """
    Splits code into chunks by double newlines. Each chunk must be at least min_chunk_length characters.
    """
    chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) >= min_chunk_length]
    return chunks

def load_dataset_descriptions_from_folder(folder_path):
    """
    Loads text files (.txt or .md) from the given folder and returns a list of dataset description items.
    """
    corpus = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        description = f.read().strip()
                    corpus.append({
                        "text": description,
                        "metadata": {
                            "type": "dataset",
                            "source": file_path,
                            "libs": []
                        }
                    })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return corpus

def load_code_from_repo(repo_path, file_extension=".py"):
    """
    Loads code files from a repository folder, splits them into chunks, and returns a list of code items.
    """
    corpus = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                    chunks = chunk_code(code)
                    if not chunks:
                        chunks = [code]
                    for chunk in chunks:
                        corpus.append({
                            "text": chunk,
                            "metadata": {
                                "type": "code",
                                "source": file_path,
                                "libs": detect_libraries(code)
                            }
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return corpus

def load_full_corpus(dataset_folder, repo_folder):
    """
    Loads the full corpus by combining dataset descriptions and code from repositories.
    """
    corpus = []
    corpus.extend(load_dataset_descriptions_from_folder(dataset_folder))
    for item in os.listdir(repo_folder):
        repo_path = os.path.join(repo_folder, item)
        if os.path.isdir(repo_path):
            corpus.extend(load_code_from_repo(repo_path))
    return corpus
