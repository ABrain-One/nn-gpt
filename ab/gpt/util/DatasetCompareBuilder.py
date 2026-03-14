#  generates the actual prompt data: Phase 1 of Cross-dataset adapatation 
import json
import pandas as pd
import ab.nn.api as lemur
from pathlib import Path
from itertools import combinations

class DatasetCompareBuilder:
    """
    Building training pairs for Phase 1:
    Same model, two different datasets, asking LLM 'which performs better?'
    """

    def __init__(self, meta_path='ab/gpt/conf/dataset_meta.json', epoch=5):
        self.epoch = epoch
        
        # Load dataset metadata
        with open(meta_path) as f:
            self.dataset_meta = json.load(f)
        
        # Load and prepare LEMUR data
        self._prepare_data()

    def _prepare_data(self):
        """Fetch data from LEMUR and compute normalization"""
        print(f"[INFO] Fetching data from LEMUR for epoch {self.epoch}...")
        df = lemur.data(only_best_accuracy=False, task='img-classification')
        df_epoch = df[df['epoch'] == self.epoch].copy()

        # Best accuracy per dataset (normalization baseline)
        self.best_per_dataset = df_epoch.groupby('dataset')['accuracy'].max()
        
        # Normalize
        df_epoch['normalized_acc'] = df_epoch.apply(
            lambda row: row['accuracy'] / self.best_per_dataset[row['dataset']], axis=1
        )

        # Keep best result per (model, dataset) combination
        self.model_best = df_epoch.groupby(['nn', 'dataset']).agg(
            normalized_acc=('normalized_acc', 'max'),
            nn_code=('nn_code', 'first')
        ).reset_index()

        print(f"[INFO] Total (model, dataset) combinations: {len(self.model_best)}")

    def build_pairs(self):
        """
        Build all training pairs:
        Each pair = same model, two different datasets
        Returns list of dicts ready for prompt injection
        """
        pairs = []
        
        # Group by model
        grouped = self.model_best.groupby('nn')
        
        for model_name, group in grouped:
            datasets = group['dataset'].tolist()
            nn_code = group['nn_code'].iloc[0]
            
            # Skip models with only 1 dataset
            if len(datasets) < 2:
                continue
            
            # Skip models whose datasets are not in metadata
            valid_datasets = [d for d in datasets if d in self.dataset_meta]
            if len(valid_datasets) < 2:
                continue

            # Generate all dataset pairs for this model
            for d1, d2 in combinations(valid_datasets, 2):
                acc1 = group[group['dataset'] == d1]['normalized_acc'].values[0]
                acc2 = group[group['dataset'] == d2]['normalized_acc'].values[0]
                meta1 = self.dataset_meta[d1]
                meta2 = self.dataset_meta[d2]

                # Ground truth answer
                better_dataset = d1 if acc1 >= acc2 else d2

                pairs.append({
                    # Model info
                    'nn': model_name,
                    'nn_code': nn_code,
                    'epoch': self.epoch,

                    # Dataset 1
                    'dataset_1': d1,
                    'norm_acc_1': round(acc1, 4),
                    'num_train_images_1': meta1['num_train_images'],
                    'img_size_1': meta1['img_size'],
                    'num_channels_1': meta1['num_channels'],
                    'num_classes_1': meta1['num_classes'],

                    # Dataset 2
                    'dataset_2': d2,
                    'norm_acc_2': round(acc2, 4),
                    'num_train_images_2': meta2['num_train_images'],
                    'img_size_2': meta2['img_size'],
                    'num_channels_2': meta2['num_channels'],
                    'num_classes_2': meta2['num_classes'],

                    # Ground truth
                    'answer': better_dataset
                })

        print(f"[INFO] Total training pairs built: {len(pairs)}")
        return pairs

    def preview(self, n=2):
        """Print n example prompts to verify correctness"""
        pairs = self.build_pairs()
        
        for i, pair in enumerate(pairs[:n]):
            print(f"\n{'='*70}")
            print(f"PAIR {i+1}: {pair['nn']} | {pair['dataset_1']} vs {pair['dataset_2']}")
            print(f"{'='*70}")
            print(f"Normalized acc → {pair['dataset_1']}: {pair['norm_acc_1']}  |  {pair['dataset_2']}: {pair['norm_acc_2']}")
            print(f"CORRECT ANSWER: {pair['answer']}")
            print(f"\nPROMPT PREVIEW:")
            print(f"A neural network was trained on two different datasets for {pair['epoch']} epochs.")
            print(f"\nDataset 1: {pair['dataset_1']}")
            print(f"  - Training images: {pair['num_train_images_1']}")
            print(f"  - Image size: {pair['img_size_1']}x{pair['img_size_1']}")
            print(f"  - Channels: {pair['num_channels_1']}")
            print(f"  - Classes: {pair['num_classes_1']}")
            print(f"  - Normalized accuracy: {pair['norm_acc_1']}")
            print(f"\nDataset 2: {pair['dataset_2']}")
            print(f"  - Training images: {pair['num_train_images_2']}")
            print(f"  - Image size: {pair['img_size_2']}x{pair['img_size_2']}")
            print(f"  - Channels: {pair['num_channels_2']}")
            print(f"  - Classes: {pair['num_classes_2']}")
            print(f"  - Normalized accuracy: {pair['norm_acc_2']}")
            print(f"\nQuestion: On which dataset does this model perform better?")
            print(f"Expected answer: {pair['answer']}")


# Quick test
if __name__ == '__main__':
    builder = DatasetCompareBuilder(epoch=5)
    builder.preview(n=3)
