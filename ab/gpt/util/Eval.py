import os
import re
import ast

import ab.nn.api as api
from ab.nn.util.Util import uuid4
from ab.gpt.util.Util import read_py_file_as_string
import ab.nn.api as nn_dataset


class Eval:
    def __init__(self, model_source_package: str, task='img-classification', dataset='cifar-10', metric='acc', prm=None, save_to_db=False, prefix=None, save_path=None, use_ast_validation=None):
        """
        Evaluates a given model on a specified dataset for classification
        :param model_source_package: The package name of the model to evaluate
        :param task: The task to evaluate the model on
        :param dataset: The dataset to evaluate the model on
        :param metric: The metric to evaluate the model on
        :param prm: The parameters to evaluate the model on
        :param save_to_db: Whether to save the results to the database
        :param use_ast_validation: None = auto-detect from prefix; True = AST; False = legacy string counting
        """
        if prm is None:
            prm = {'lr': 0.01, 'batch': 10, 'dropout': 0.2, 'momentum': 0.9,
                   'transform': 'norm_256_flip', 'epoch': 1}
        self.model_package = model_source_package
        self.task = task
        self.dataset = dataset
        self.metric = metric
        self.prm = prm
        self.save_to_db = save_to_db
        self.prefix = prefix
        self.save_path = save_path
        
        if use_ast_validation is None:
            self.use_ast_validation = prefix is not None and 'delta' in str(prefix).lower()
        else:
            self.use_ast_validation = use_ast_validation

    def evaluate(self, nn_file):
        os.listdir(self.model_package)
        code = read_py_file_as_string(nn_file)
        if not code or not code.strip():
            raise Exception(f'Code is missing.')
        sup_prm = 'supported_hyperparameters'
        for fn in {sup_prm, 'train_setup', 'learn'}:
            if not re.match(r'[\s\S]*\s+def\s' + re.escape(fn) + r'\(.*', code):
                raise Exception(f'The NN code lacks the required function \'{fn}\'.')

        tree = ast.parse(code)
        prm_keys = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == sup_prm:
                if isinstance(node.body[0], ast.Return):
                    return_node = node.body[0].value
                    prm_keys = ast.literal_eval(return_node)
        
        if self.use_ast_validation:
            for prm_key in prm_keys:
                param_used = False
                for node in ast.walk(tree):
                    # Check for prm['key'] pattern (subscript)
                    if isinstance(node, ast.Subscript):
                        if isinstance(node.value, ast.Name) and node.value.id == 'prm':
                            if isinstance(node.slice, ast.Constant) and node.slice.value == prm_key:
                                param_used = True
                                break
                    # Check for prm.get('key') pattern (method call)
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
                            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'prm':
                                if len(node.args) > 0 and isinstance(node.args[0], ast.Constant) and node.args[0].value == prm_key:
                                    param_used = True
                                    break
                if not param_used:
                    raise Exception(f'The param \'{prm_key}\' is not used in the code.')
        else:
            for prm_key in prm_keys:
                if code.count('"' + prm_key + '"') + code.count("'" + prm_key + "'") < 2:
                    raise Exception(f'The param \'{prm_key}\' is not used in the code.')

        nn_dataset.data.cache_clear()
        df = nn_dataset.data()
        ids_list = df["nn_id"].unique().tolist() if "nn_id" in df.columns else []
        new_checksum = uuid4(code)
        if new_checksum not in ids_list:
            return api.check_nn(code, self.task, self.dataset, self.metric, self.prm, self.save_to_db, self.prefix, self.save_path)
        else:
            raise Exception(f'NN already exists (checksum: {new_checksum}). Skipping API call.')

    def get_args(self):
        return {
            'model_package': self.model_package,
            'task': self.task,
            'dataset': self.dataset,
            'metric': self.metric,
            'prm': self.prm,
        }
