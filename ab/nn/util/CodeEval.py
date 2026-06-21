import ast
import importlib.util
import json
import logging
import os
import pprint
import subprocess
import sys
from os import makedirs

import torch
from radon.complexity import cc_visit
from ab.nn.util.Const import nn_dir, out_dir, default_nn_path
from ab.nn.util.Util import torch_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_python_files(directory):
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

# Static Analysis using Pylint
def run_pylint(file_path):
    max_line_length = 120
    disable_errors = ['C0116', 'C0115', 'C0114', 'C0304', 'C0303', 'C0305', 'C0325']  # For docstrings and missing newline and whitespace
    command = [
        sys.executable, '-m', 'pylint', file_path,
        '--output-format=json',
        f'--max-line-length={max_line_length}'
    ]
    if disable_errors:
        disable_str = ",".join(disable_errors)
        command.append(f'--disable={disable_str}')

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        logging.warning("Pylint execution failed for %s: %s", file_path, e)
        return []

    if result.stdout:
        try:
            errors = json.loads(result.stdout)
            return errors
        except json.JSONDecodeError as e:
            print(f"JSON decode error in Pylint output for {file_path}: {e}")
            return []

    if result.returncode not in (0, 1, 2, 4, 8, 16, 32):
        logging.warning("Unexpected pylint return code %s for %s", result.returncode, file_path)
    return []

# Complexity Analysis
def analyze_complexity(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        complexity = cc_visit(code)
        complexity_report = []
        for block in complexity:
            block_type = getattr(block, 'type', getattr(block, 'kind', 'unknown'))
            complexity_report.append({
                'name': block.name,
                'type': block_type,
                'complexity': block.complexity,
                'lineno': block.lineno,
                'endline': block.endline
            })
        return complexity_report
    except Exception as e:
        print(f"Error analyzing complexity for {file_path}: {e}")
        return []

# Docstrings and Comments Checker
class DocstringChecker(ast.NodeVisitor):
    def __init__(self):
        self.missing_docstrings = []

    def visit_FunctionDef(self, node):
        if not ast.get_docstring(node):
            self.missing_docstrings.append(f"Function '{node.name}' is missing a docstring")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if not ast.get_docstring(node):
            self.missing_docstrings.append(f"Class '{node.name}' is missing a docstring")
        self.generic_visit(node)

def check_docstrings(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        checker = DocstringChecker()
        checker.visit(tree)
        return checker.missing_docstrings
    except Exception as e:
        print(f"Error checking docstrings for {file_path}: {e}")
        return []

# Dynamic Evaluation: Attempt to Run the Code
def dynamic_evaluation(file_path, class_name='Net'):
    try:
        # Dynamically import the module
        spec = importlib.util.spec_from_file_location("model_module", file_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Get the model class
        ModelClass = getattr(model_module, class_name)

        # Example parameters
        in_shape = (1, 3, 224, 224)  # Assuming input is an RGB image of size 224x224
        out_shape = (10,)  # Assuming there are 10 classes
        prm = {
            'lr': 0.01,
            'momentum': 0.9,
            'dropout': 0.5
        }

        # Set up the device
        device = torch_device()

        # Instantiate the model
        model = ModelClass(in_shape, out_shape, prm, device)
        logging.info("Model instantiation succeeded.")

        model.to(device)
        model.train_setup(prm)
        logging.info(f"The training environment is set up successfully, using the following device: {device}.")

        # Create simulated data
        batch_size = 8
        num_batches = 2
        input_tensor = torch.randn(batch_size, *in_shape[1:]).to(device)
        labels = torch.randint(0, out_shape[0], (batch_size,)).to(device)
        train_data = [(input_tensor, labels) for _ in range(num_batches)]
        logging.info("Simulated training data created successfully.")

        # Execute training step
        model.learn(train_data)
        logging.info("Training step executed successfully.")

        return {
            'success': True,
            'output': 'Success'
        }

    except Exception as e:
        logging.error(f"Evaluation Failed: {e}")
        return {
            'success': False,
            'output': str(e)
        }

# Calculate Score
def calculate_score(pylint_issues, complexity_report, dynamic_success, missing_docstrings=None):
    score = 100
    # Pylint scoring
    for issue in pylint_issues:
        if issue['type'] == 'error':
            score -= 5
        elif issue['type'] == 'warning':
            score -= 2
        elif issue['type'] == 'convention':
            score -= 1
        elif issue['type'] == 'refactor':
            score -= 1

    # Complexity scoring
    for item in complexity_report:
        if item['complexity'] > 10:
            score -= 1

    # Docstrings
    # if missing_docstrings:
    #     score -= len(missing_docstrings) * 1

    # Dynamic evaluation scoring
    if not dynamic_success:
        score -= 10

    if score < 0:
        score = 0
    return score / 100.0

# Evaluate Code Quality for a Single File
def evaluate_code_quality(file_path):
    success = False
    try:
        report = {}

        # Static analysis
        pylint_issues = run_pylint(file_path)
        report['pylint'] = pylint_issues

        # Complexity analysis
        complexity_report = analyze_complexity(file_path)
        report['complexity'] = complexity_report

        # Dynamic evaluation
        dynamic_result = dynamic_evaluation(file_path, class_name='Net')
        dynamic_success = bool(dynamic_result.get('success', False))
        dynamic_output = dynamic_result.get('output', '')
        report['dynamic_evaluation'] = {
            'success': dynamic_success,
            'output': dynamic_output
        }
        # Calculate score
        score = calculate_score(pylint_issues, complexity_report, dynamic_success)  # Docstrings
        report['score'] = score
        success = True
    except Exception as e:
        print(f"Error evaluating {file_path}: {e}")
        report = {
            'error': str(e)
        }
    return success, report

# Evaluate Code Quality for an Entire Directory
def evaluate_directory_code_quality(directory):
    report = {}
    python_files = get_python_files(directory)
    report['files'] = {}
    report['total_score'] = 0
    report['max_score'] = len(python_files) * 100
    report['average_score'] = 0

    for file_path in python_files:
        print(f"Evaluating {file_path}...")
        success, file_report = evaluate_code_quality(file_path)
        report['files'][file_path] = file_report
        report['total_score'] += file_report['score']

    # Calculate average score
    if len(python_files) > 0:
        report['average_score'] = report['total_score'] / len(python_files)
    else:
        report['average_score'] = 0

    return report

def evaluate_single_file(file_path):
    print(f"Evaluating {file_path}...")
    success, report = evaluate_code_quality(file_path)
    if not isinstance(report, dict):
        return {'score': 0.0, 'error': 'invalid quality report'}
    if 'score' not in report:
        report['score'] = 0.0
    if not success and 'error' not in report:
        report['error'] = 'code quality evaluation failed'
    return report

# Main Function
if __name__ == "__main__":
    # Single file evaluation
    res = evaluate_single_file(default_nn_path)
    pprint.pprint(res.get('score'))

    quality_report = evaluate_directory_code_quality(nn_dir)
    # # Print detailed report
    pprint.pprint(quality_report)

    # # Save as JSON file
    makedirs(out_dir, exist_ok=True)
    with open(out_dir / 'quality_report.json', 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, ensure_ascii=False, indent=4)

    print("Quality report saved to quality_report.json")
