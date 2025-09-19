"""
Dataset Loader / Registry

Provides a centralized registry for datasets (like the ModelLoader pattern).
It discovers dataset subdirectories under the repo `dataset/` folder and
exposes a simple API to get dataset paths, register custom dataset handlers,
and list available datasets for CLI integration.

Usage:
    from src.loader.dataset_loader import dataset_loader
    dataset_dir = dataset_loader.get_dataset_dir('cmohs')
    choices = dataset_loader.get_dataset_choices()
    handler = dataset_loader.get_handler('cmohs', config)

This keeps dataset access non-hardcoded and extensible.
"""

import os
import glob
import yaml
from typing import Dict, Optional, Any, List, Tuple
from .handlers import get_handler_for_dataset, BaseDatasetHandler


class DatasetLoader:
    """Central dataset registry and loader.

    On init it discovers subdirectories in the repo `dataset/` folder and
    registers them as available datasets. Additional datasets can be
    registered with `register_dataset`.
    """

    def __init__(self, repo_root: Optional[str] = None):
        # Determine repository root (two levels up from this file: src/loader)
        if repo_root:
            self.repo_root = os.path.abspath(repo_root)
        else:
            self.repo_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', '..'))

        self.dataset_root = os.path.join(self.repo_root, 'dataset')
        self._datasets: Dict[str, Dict[str, Any]] = {}

        # Load dataset configurations
        self.dataset_configs = self._load_dataset_configs()

        # Discover datasets automatically
        self._discover_datasets()

    def _load_dataset_configs(self) -> Dict[str, Any]:
        """Load dataset configurations from datasets.yaml"""
        config_path = os.path.join(
            self.repo_root, 'src', 'config', 'datasets.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('datasets', {})
        return {}

    def _discover_datasets(self) -> None:
        """Discover datasets in the dataset/ directory"""
        if not os.path.exists(self.dataset_root):
            return

        for item in os.listdir(self.dataset_root):
            item_path = os.path.join(self.dataset_root, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Skip backup directories
                if item == 'all':
                    continue

                # Analyze dataset structure
                usable, dataset_type = self._analyze_dataset_structure(
                    item_path)

                # Get metadata from config if available
                metadata = self.dataset_configs.get(item, {})

                self._datasets[item] = {
                    'path': item_path,
                    'type': dataset_type,
                    'usable': usable,
                    'metadata': metadata,
                    'config': metadata.get('preprocessing', {})
                }

    def _analyze_dataset_structure(self, dataset_dir: str) -> Tuple[bool, str]:
        """Analyze directory structure to determine dataset type"""
        files = os.listdir(dataset_dir)

        # Check for profile.txt (text sensors dataset like cmohs)
        if 'profile.txt' in files:
            # Count sensor text files (exclude long descriptive files)
            sensor_files = [f for f in files if f.endswith('.txt') and 
                          f not in ['profile.txt', 'description.txt', 'documentation.txt']]
            if len(sensor_files) >= 3:
                return True, 'text_sensors'

        # Check for single CSV file
        csv_files = [f for f in files if f.endswith('.csv')]
        if len(csv_files) == 1:
            return True, 'csv'

        # Check for multiple CSV files
        if len(csv_files) > 1:
            return True, 'multi_csv'

        # Check for text data files (space-separated)
        txt_files = [f for f in files if f.endswith(
            '.txt') and 'train' in f.lower() or 'test' in f.lower()]
        if len(txt_files) >= 2:
            return True, 'text_data'

        # Default to empty if no recognizable pattern
        return False, 'empty'

    def get_dataset_choices(self) -> List[str]:
        """Get list of usable dataset names for CLI choices"""
        return [name for name, info in self._datasets.items() if info.get('usable')]

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get full information about a dataset"""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found")
        return self._datasets[name]

    def get_dataset_dir(self, name: str) -> str:
        """Get the directory path for a dataset"""
        return self.get_dataset_info(name)['path']

    def register_dataset(self, name: str, path: str, dataset_type: str = 'custom', **metadata) -> None:
        """Register a custom dataset"""
        self._datasets[name] = {
            'path': os.path.abspath(path),
            'type': dataset_type,
            'usable': True,
            'metadata': metadata
        }

    def get_handler(self, name: str, global_config: Optional[Dict[str, Any]] = None) -> BaseDatasetHandler:
        """Get a dataset handler for the specified dataset"""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found")

        dataset_dir = self._datasets[name]['path']
        dataset_config = self._datasets[name].get('config', {})

        # Merge global config with dataset-specific config
        if global_config:
            merged_config = {**global_config, **dataset_config}
        else:
            merged_config = dataset_config

        return get_handler_for_dataset(dataset_dir, merged_config)

    def list_datasets(self, verbose: bool = False) -> None:
        """List all discovered datasets"""
        print("Available datasets:")
        for name, info in self._datasets.items():
            usable_flag = ' (usable)' if info.get('usable') else ' (ignored)'
            print(f"- {name}{usable_flag}: {info['path']}")
            if verbose:
                print(f"  type: {info.get('type')}")
                print(f"  metadata: {info.get('metadata')}")


# Global singleton instance
dataset_loader = DatasetLoader()


def get_dataset(name: str) -> Dict[str, Any]:
    """Get dataset info using the global instance"""
    return dataset_loader.get_dataset_info(name)


def get_dataset_dir(name: str) -> str:
    """Get dataset directory using the global instance"""
    return dataset_loader.get_dataset_dir(name)


if __name__ == '__main__':
    print('Dataset Loader Demo')
    dataset_loader.list_datasets(verbose=True)
