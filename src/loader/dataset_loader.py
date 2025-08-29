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

This keeps dataset access non-hardcoded and extensible.
"""

import os
import glob
from typing import Dict, Optional, Any, List, Tuple


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

        # Discover datasets automatically
        self._discover_datasets()

    def _discover_datasets(self) -> None:
        """Discover subdirectories under dataset/ and register them"""
        if not os.path.isdir(self.dataset_root):
            return

        # Some directories in dataset/ are backups (for example 'all').
        # Only register folders that look like real datasets (contain profile.txt,
        # or contain non-empty CSV/TXT sensor files). This avoids exposing
        # archive/backup folders as processable datasets.
        ignore_dirs = {'all', '.gitkeep', '__pycache__'}

        for entry in sorted(os.listdir(self.dataset_root)):
            if entry in ignore_dirs:
                continue

            path = os.path.join(self.dataset_root, entry)
            if not os.path.isdir(path):
                continue

            # Determine if folder looks like a usable dataset
            usable, dataset_type = self._is_dataset_usable(path)

            # Register with minimal metadata
            self._datasets[entry] = {
                'name': entry,
                'path': path,
                'type': dataset_type,
                'usable': usable,
                'metadata': {}
            }

    def register_dataset(self, name: str, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register or override a dataset entry.

        Args:
            name: dataset key name
            path: absolute or relative path to dataset folder
            metadata: optional dict of metadata
        """
        abs_path = os.path.abspath(path)
        self._datasets[name] = {
            'name': name,
            'path': abs_path,
            'type': 'custom',
            'usable': True,
            'metadata': metadata or {}
        }

    def get_available_datasets(self) -> List[str]:
        """Return list of registered dataset names"""
        return list(self._datasets.keys())

    def get_usable_datasets(self) -> List[str]:
        """Return list of datasets that look processable (usable==True)"""
        return [n for n, v in self._datasets.items() if v.get('usable', False)]

    def get_dataset_dir(self, name: str) -> str:
        """Get the absolute dataset directory for a registered dataset name.

        Raises KeyError if not found.
        """
        if name not in self._datasets:
            raise KeyError(
                f"Dataset '{name}' is not registered. Available: {self.get_available_datasets()}")
        return self._datasets[name]['path']

    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Return the dataset registry entry"""
        return self._datasets.get(name, {})

    def list_datasets(self, verbose: bool = False) -> None:
        print("Available datasets:")
        for name, info in self._datasets.items():
            usable_flag = ' (usable)' if info.get('usable') else ' (ignored)'
            print(f"- {name}{usable_flag}: {info['path']}")
            if verbose:
                print(f"  type: {info.get('type')}")
                print(f"  metadata: {info.get('metadata')}")

    def get_dataset_choices(self) -> List[str]:
        """Return choices suitable for CLI argument parsing"""
        # By default, only return usable datasets so CLI choices don't include backup folders
        return self.get_usable_datasets()

    def _is_dataset_usable(self, path: str) -> Tuple[bool, str]:
        """Heuristic checks to decide if a folder under dataset/ is processable.

        Returns (usable: bool, dataset_type: str)
        dataset_type is a short hint: 'folder_profile', 'single_csv', 'text_sensors', or 'empty'.
        """
        # Check for profile.txt (like cmohs)
        profile_path = os.path.join(path, 'profile.txt')
        if os.path.isfile(profile_path):
            return True, 'folder_profile'

        # Check for at least one non-empty CSV file (equipmentad, mlclassem, smartmd)
        csv_files = [f for f in os.listdir(path) if f.lower().endswith('.csv')]
        for f in csv_files:
            full = os.path.join(path, f)
            try:
                if os.path.getsize(full) > 0:
                    return True, 'single_csv'
            except OSError:
                continue

        # Check for multiple non-empty .txt sensor files
        txt_files = [f for f in os.listdir(path) if f.lower().endswith('.txt')]
        valid_txts = 0
        for f in txt_files:
            if f.lower() in ('description.txt', 'documentation.txt', 'readme.txt'):
                continue
            full = os.path.join(path, f)
            try:
                if os.path.getsize(full) > 0:
                    valid_txts += 1
            except OSError:
                continue

        if valid_txts >= 1:
            return True, 'text_sensors'

        # No usable files discovered
        return False, 'empty'


# Global singleton instance
dataset_loader = DatasetLoader()


def get_dataset(name: str) -> Dict[str, Any]:
    return dataset_loader.get_dataset_info(name)


def get_dataset_dir(name: str) -> str:
    return dataset_loader.get_dataset_dir(name)


if __name__ == '__main__':
    print('Dataset Loader Demo')
    dataset_loader.list_datasets(verbose=True)
