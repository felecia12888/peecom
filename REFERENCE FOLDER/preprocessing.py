# preprocessing.py
import os
import logging
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from torchvision import transforms
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Optional, Union, cast
from collections.abc import Sized
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
import cv2
import yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Config Loader
# -----------------------------------------------------------------------------
def load_config(config_path="config.yaml"):
    """
    Loads a YAML configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_preprocessing_config(config_path: str = "./loader/config.yaml") -> dict:
    config = load_config(config_path)
    return config.get("data", {})


# Add this function after the config loading functions
def get_default_transforms(preproc_config: dict = None):
    """
    Returns a dictionary of transforms for 'train', 'val', and 'test' sets.
    If preproc_config is provided (a dict from YAML), its parameters are used.
    """
    # Use config to determine resize value; default to [224, 224]
    resize = preproc_config.get(
        "resize", [224, 224]) if preproc_config else [224, 224]
    train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])
    test = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])
    return {"train": train, "val": val, "test": test}


# -----------------------------------------------------------------------------
# Additional Preprocessing Functions
# -----------------------------------------------------------------------------
def apply_hu_window(image: Image.Image, window: Tuple[int, int] = (-1000, 400)) -> Image.Image:
    """
    For CT scans: apply Hounsfield unit (HU) windowing.
    """
    image_np = np.array(image).astype(np.float32)
    lower, upper = window
    image_np = np.clip(image_np, lower, upper)
    # Normalize to [0, 1]
    image_np = (image_np - lower) / (upper - lower)
    # Convert back to 8-bit image
    return Image.fromarray((image_np * 255).astype(np.uint8))


def apply_clahe(image: Image.Image, clip_limit=2.0, tile_grid_size=(8, 8)) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement.
    """
    image_np = np.array(image)
    # If image is grayscale
    if len(image_np.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)
        image_clahe = clahe.apply(image_np)
        return Image.fromarray(image_clahe)
    else:
        # For color images, convert to LAB and apply CLAHE on the L channel.
        image_lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(image_lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        image_lab = cv2.merge((l, a, b))
        image_clahe = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(image_clahe)


def apply_median_filter(image: Image.Image, kernel_size=3) -> Image.Image:
    """
    Apply a median filter to remove salt-and-pepper noise.
    """
    image_np = np.array(image)
    filtered = cv2.medianBlur(image_np, kernel_size)
    return Image.fromarray(filtered)


def apply_gaussian_filter(image: Image.Image, sigma=1.0) -> Image.Image:
    """
    Apply a Gaussian filter to reduce noise.
    """
    image_np = np.array(image)
    filtered = cv2.GaussianBlur(image_np, (0, 0), sigma)
    return Image.fromarray(filtered)


def apply_histogram_equalization(image: Image.Image) -> Image.Image:
    """
    Apply histogram equalization to a grayscale image.
    """
    gray = image.convert("L")
    image_np = np.array(gray)
    eq = cv2.equalizeHist(image_np)
    return Image.fromarray(eq)


def apply_padding(image: Image.Image, padding_config: dict) -> Image.Image:
    """
    Apply padding based on configuration.
    """
    if not padding_config.get('enabled', False):
        return image

    strategy = padding_config.get('strategy', 'symmetric')
    if strategy == 'symmetric':
        return ImageOps.expand(image, border=padding_config.get('value', 10), fill=0)
    elif strategy == 'constant':
        return transforms.Pad(padding_config.get('value', 0))(image)
    return image


def apply_brightness_adjustment(image: Image.Image, factor: float = 1.2) -> Image.Image:
    """
    Adjust image brightness.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def apply_padding_with_strategy(image: Image.Image, padding_config: dict) -> Image.Image:
    """
    Enhanced padding function that supports multiple strategies from config:
    - symmetric: Equal padding on all sides
    - constant: Padding with constant value
    - reflect: Mirror padding
    """
    if not padding_config.get('enabled', False):
        return image

    strategy = padding_config.get('strategy', 'symmetric')
    pad_value = padding_config.get('value', 0)

    if strategy == 'symmetric':
        # Calculate padding to make image square if pad_to_square is True
        if padding_config.get('pad_to_square', False):
            w, h = image.size
            max_dim = max(w, h)
            delta_w = max_dim - w
            delta_h = max_dim - h
            padding = (delta_w//2, delta_h//2, delta_w -
                       (delta_w//2), delta_h-(delta_h//2))
            return ImageOps.expand(image, padding, fill=pad_value)
        return ImageOps.expand(image, border=pad_value, fill=0)
    elif strategy == 'reflect':
        return ImageOps.expand(image, border=pad_value, fill=ImageOps.REFLECT)
    else:  # constant
        return ImageOps.expand(image, border=pad_value, fill=pad_value)


def apply_image_enhancement(image: Image.Image, enhancement_config: dict) -> Image.Image:
    """
    Apply multiple image enhancements based on config
    """
    if enhancement_config.get('brightness_adjustment', False):
        factor = enhancement_config.get('brightness_factor', 1.2)
        image = ImageEnhance.Brightness(image).enhance(factor)

    if enhancement_config.get('contrast_enhancement', False):
        factor = enhancement_config.get('contrast_factor', 1.5)
        image = ImageEnhance.Contrast(image).enhance(factor)

    return image


def apply_noise_reduction(image: Image.Image, config: dict) -> Image.Image:
    """
    Apply noise reduction based on config settings
    """
    method = config.get('noise_reduction_method', 'gaussian')
    if method == 'gaussian':
        return image.filter(ImageFilter.GaussianBlur(radius=1))
    elif method == 'median':
        return image.filter(ImageFilter.MedianFilter(size=3))
    return image


class DatasetPreprocessor:
    """New class to handle dataset-specific preprocessing"""

    def __init__(self, dataset_name: str, config: dict):
        self.dataset_name = dataset_name
        self.config = config
        self.dataset_config = self._get_dataset_config()
        self.structure_type = self._get_structure_type()

    def _get_dataset_config(self):
        return next((d for d in self.config['data_key']
                    if d['name'] == self.dataset_name), None)

    def _get_structure_type(self):
        return self.dataset_config['structure'].get('type', 'standard')

    def preprocess(self, data_path: Path, mode: str) -> Dataset:
        """Main preprocessing pipeline"""
        transforms_list = self._build_transform_pipeline(mode)
        if self.structure_type == "class_based":
            return self._preprocess_class_based(data_path, transforms_list)
        return preprocess_dataset(str(data_path), self.dataset_name, mode)

    def _build_transform_pipeline(self, mode: str):
        """Build transforms with fallback to default transforms"""
        config = load_preprocessing_config()
        ds_cfg = config.get(self.dataset_name, {})
        common_cfg = config.get('common_settings', {})

        # Get default transforms as fallback
        default_transforms = get_default_transforms(common_cfg)

        # If no special processing needed, return default transforms
        if not ds_cfg.get('preprocessing') and not ds_cfg.get('augmentation'):
            return default_transforms[mode]

        # Otherwise build custom transform pipeline
        transform_list = []

        # 1. Padding (with enhanced strategies)
        padding_config = {
            **common_cfg.get('padding', {}), **ds_cfg.get('padding', {})}
        if padding_config.get('enabled', False):
            transform_list.append(
                transforms.Lambda(
                    lambda img: apply_padding_with_strategy(img, padding_config))
            )

        # 2. Channel conversion
        if ds_cfg.get("conversion", "") == "convert_to_3_channel":
            transform_list.append(
                transforms.Lambda(lambda img: img.convert("RGB"))
            )

        # 3. Base resize
        resize_size = ds_cfg.get(
            "resize", common_cfg.get("resize", [224, 224]))
        transform_list.append(transforms.Resize(resize_size))

        # 4. Training mode specific transforms
        if mode == "train":
            # Apply preprocessing before augmentation
            preproc_cfg = ds_cfg.get('preprocessing', {})
            if preproc_cfg.get('clahe', False):
                transform_list.append(
                    transforms.Lambda(lambda img: apply_clahe(img))
                )
            if preproc_cfg.get('remove_noise', False):
                transform_list.append(
                    transforms.Lambda(
                        lambda img: apply_noise_reduction(img, preproc_cfg))
                )

            # Augmentation pipeline
            aug_cfg = ds_cfg.get('augmentation', {})
            if aug_cfg:
                transform_list.append(
                    transforms.Lambda(
                        lambda img: apply_image_enhancement(img, aug_cfg))
                )
                # Add random transforms for training
                if aug_cfg.get('random_horizontal_flip', True):
                    transform_list.append(transforms.RandomHorizontalFlip())
                if aug_cfg.get('random_rotation', False):
                    transform_list.append(transforms.RandomRotation(10))

        # 5. To Tensor and Normalization
        transform_list.append(transforms.ToTensor())
        if "normalization" in ds_cfg:
            norm_cfg = ds_cfg["normalization"]
            transform_list.append(transforms.Normalize(
                mean=norm_cfg['mean'],
                std=norm_cfg['std']
            ))

        return transforms.Compose(transform_list)


def build_transforms(dataset_name: str, mode: str = "train"):
    """Updated to handle all dataset structures"""
    config = load_preprocessing_config()
    preprocessor = DatasetPreprocessor(dataset_name, config)
    return preprocessor._build_transform_pipeline(mode)


def preprocess_dataset(dataset_dir: str, dataset_name: str, mode: str = "train") -> Dataset:
    """Updated to handle different dataset structures"""
    config = load_preprocessing_config()
    preprocessor = DatasetPreprocessor(dataset_name, config)
    return preprocessor.preprocess(Path(dataset_dir), mode)


# Add new helper functions for specific preprocessing tasks
def process_metadata(metadata_file: Path) -> dict:
    """Process dataset metadata files"""
    if metadata_file.suffix == '.xlsx':
        return pd.read_excel(metadata_file)
    return pd.read_csv(metadata_file)


def validate_dataset_structure(dataset_path: Path, expected_structure: dict) -> bool:
    """Validate dataset folder structure"""
    if not dataset_path.exists():
        return False

    structure_type = expected_structure.get('type', 'standard')

    if structure_type == 'class_based':
        return all(
            (dataset_path / class_name).exists()
            for class_name in expected_structure['classes']
        )

    required_dirs = {
        'standard': ['train', 'val', 'test'],
        'train_test': ['train', 'test'],
        'train_valid_test': ['train', 'valid', 'test']
    }

    return all(
        (dataset_path / dir_name).exists()
        for dir_name in required_dirs.get(structure_type, [])
    )


# Update split_dataset function to handle different structures
def split_dataset(
    dataset: Union[Dataset, Tuple[Dataset, ...], Sized],
    split_ratios: List[float] = None,
    structure_type: str = 'standard'
) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """Enhanced split function that handles different dataset structures"""
    if dataset is None or len(dataset) == 0:
        return None, None, None

    if isinstance(dataset, tuple):
        if all(isinstance(d, (Dataset, type(None))) for d in dataset):
            return cast(Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]], dataset)
        raise ValueError(
            "All elements must be torch.utils.data.Dataset instances or None.")

    if not split_ratios:
        split_ratios = [0.7, 0.15, 0.15]  # default ratios

    if structure_type == 'train_test':
        # Split train into train/val
        train_size = split_ratios[0] / (split_ratios[0] + split_ratios[1])
        train_data, val_data = train_test_split(
            dataset, train_size=train_size, random_state=42)
        return train_data, val_data, None

    # Standard three-way split
    train_size = int(split_ratios[0] * len(dataset))
    val_size = int(split_ratios[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    if train_size > 0 and val_size > 0 and test_size > 0:
        return random_split(dataset, [train_size, val_size, test_size])

    raise ValueError("Dataset too small for splitting")


# -----------------------------------------------------------------------------
# Outlier Removal Methods
# -----------------------------------------------------------------------------
def remove_outliers_isolation_forest(X, contamination=0.1):
    n_samples, channels, height, width = X.shape
    X_reshaped = X.reshape(n_samples, -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    iso_forest = IsolationForest(contamination=contamination)
    iso_forest.fit(X_scaled)
    outliers = iso_forest.predict(X_scaled)
    outlier_indices = np.where(outliers == -1)[0]
    X_cleaned = np.delete(X, outlier_indices, axis=0)
    return X_cleaned


def remove_outliers_lof(X, n_neighbors=20, contamination=0.1):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                             contamination=contamination)
    outliers = lof.fit_predict(X_scaled)
    outlier_indices = np.where(outliers == -1)[0]
    X_cleaned = np.delete(X, outlier_indices, axis=0)
    return X_cleaned


def remove_outliers_dbscan(X, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    outlier_indices = np.where(clusters == -1)[0]
    X_cleaned = np.delete(X, outlier_indices, axis=0)
    return X_cleaned


# -----------------------------------------------------------------------------
# Dataset Splitting and Sampler
# -----------------------------------------------------------------------------
def get_WeightedRandom_Sampler(subset_dataset, original_dataset):
    original_dataset = original_dataset.dataset if isinstance(original_dataset,
                                                              torch.utils.data.Subset) else original_dataset
    dataLoader = DataLoader(subset_dataset, batch_size=512)
    All_target = []
    for _, (_, targets) in enumerate(dataLoader):
        for i in range(targets.shape[0]):
            All_target.append(targets[i].item())
    target = np.array(All_target)
    logging.info("\nClass distribution in the dataset:")
    for i, class_name in enumerate(original_dataset.classes):
        logging.info(f"{np.sum(target == i)}: {class_name}")
    class_sample_count = np.array(
        [len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight).double()
    Sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return Sampler


def get_dataloader_target_class_number(dataLoader):
    original_dataset = dataLoader.dataset
    if isinstance(original_dataset, torch.utils.data.Subset):
        original_dataset = original_dataset.dataset
    All_target_2 = []
    for batch_idx, (inputs, targets) in enumerate(dataLoader):
        for i in range(targets.shape[0]):
            All_target_2.append(targets[i].item())
    data = np.array(All_target_2)
    unique_classes, counts = np.unique(data, return_counts=True)
    logging.info("Unique classes and their counts in the dataset:")
    for cls, count in zip(unique_classes, counts):
        logging.info(f"{count}: {original_dataset.classes[cls]}")
    return original_dataset.classes, len(original_dataset.classes)


def check_for_corrupted_images(directory, transform):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img = transform(img)
                except Exception as e:
                    logging.error(f"Corrupted image file: {img_path} - {e}")


# -----------------------------------------------------------------------------
# Dataset Preprocessing
# -----------------------------------------------------------------------------
def preprocess_dataset(train_dataset):
    # Convert dataset to a NumPy array of images
    train_data = np.array([np.array(img) for img, _ in train_dataset])
    train_labels = np.array([label for _, label in train_dataset])
    # Remove outliers using Isolation Forest (or you could choose another method)
    train_data_cleaned = remove_outliers_isolation_forest(train_data)
    # Convert cleaned images back to PIL Images and recreate dataset
    train_dataset_cleaned = [
        (transforms.ToPILImage()(img.permute(1, 2, 0).numpy().astype(np.uint8)), label)
        for img, label in train_data_cleaned
    ]
    return train_dataset_cleaned


# Add these new functions to your existing preprocessing.py file

def apply_aspect_aware_resize(image: Image.Image, target_size: tuple,
                              pad_mode: str = 'constant', pad_color: int = 0) -> Image.Image:
    """
    Resize image while preserving aspect ratio, then pad to target size.

    Args:
        image: PIL image
        target_size: (width, height) tuple
        pad_mode: 'constant' or 'reflect' for padding method
        pad_color: Fill color for padding (used with constant mode)

    Returns:
        Resized and padded image
    """
    w, h = image.size
    target_w, target_h = target_size

    # Determine scaling factor to preserve aspect ratio
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)

    # Resize the image
    resized_img = image.resize((new_w, new_h), Image.BICUBIC)

    # Create a new image with the target size and paste the resized image
    result = Image.new(image.mode, target_size, pad_color)

    # Compute position to paste (center the image)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2

    # Paste the resized image onto the padded one
    result.paste(resized_img, (paste_x, paste_y))

    return result


def detect_and_correct_grayscale(image: Image.Image, force_rgb: bool = True) -> Image.Image:
    """
    Detect if an image is grayscale (even if in RGB format) and standardize to RGB or grayscale.

    Args:
        image: PIL image
        force_rgb: If True, convert to RGB; if False, convert to grayscale when applicable

    Returns:
        Standardized image
    """
    # Check the current mode
    if image.mode == 'RGB' and force_rgb:
        return image

    if image.mode == 'L' and not force_rgb:
        return image

    # Handle different modes
    if image.mode == 'L' and force_rgb:
        return image.convert('RGB')

    if image.mode == 'RGB' and not force_rgb:
        # Check if it's actually a grayscale image in RGB format
        img_array = np.array(image)
        is_grayscale = np.all(img_array[:, :, 0] == img_array[:, :, 1]) and np.all(
            img_array[:, :, 1] == img_array[:, :, 2])
        if is_grayscale:
            return image.convert('L')

    # Handle RGBA, P, and other modes
    if image.mode == 'RGBA':
        return image.convert('RGB') if force_rgb else image.convert('L')

    if image.mode == 'P':
        return image.convert('RGB') if force_rgb else image.convert('L')

    # Default to requested format
    return image.convert('RGB') if force_rgb else image.convert('L')


def enhance_image_quality(image: Image.Image, config: dict = None) -> Image.Image:
    """
    Comprehensive image quality enhancement based on config settings.

    Args:
        image: PIL image
        config: Dictionary with enhancement settings
            - brightness_factor: float, brightness adjustment (default: 1.0)
            - contrast_factor: float, contrast adjustment (default: 1.0)
            - sharpness_factor: float, sharpness adjustment (default: 1.0)
            - auto_equalize: bool, apply histogram equalization (default: False)
            - auto_enhance: bool, apply auto-enhance (default: False)

    Returns:
        Enhanced image
    """
    if config is None:
        config = {}

    # Auto-equalize histogram if requested
    if config.get('auto_equalize', False):
        image = ImageOps.equalize(image)

    # Apply brightness adjustment if factor is not 1.0
    brightness_factor = config.get('brightness_factor')
    if brightness_factor is not None and brightness_factor != 1.0:
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

    # Apply contrast adjustment if factor is not 1.0
    contrast_factor = config.get('contrast_factor')
    if contrast_factor is not None and contrast_factor != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

    # Apply sharpness adjustment if factor is not 1.0
    sharpness_factor = config.get('sharpness_factor')
    if sharpness_factor is not None and sharpness_factor != 1.0:
        image = ImageEnhance.Sharpness(image).enhance(sharpness_factor)

    # Apply auto enhance if requested (applies all standard enhancements)
    if config.get('auto_enhance', False):
        image = ImageOps.autocontrast(image)

    return image


def apply_color_correction(image: Image.Image, config: dict = None) -> Image.Image:
    """
    Apply color correction to handle color casts and improve color balance.

    Args:
        image: PIL image
        config: Dictionary with color correction settings
            - white_balance: bool, apply white balance (default: False)
            - saturation_factor: float, saturation adjustment (default: 1.0)
            - temperature: float, color temperature adjustment (-1.0 to 1.0, default: 0)

    Returns:
        Color corrected image
    """
    if config is None:
        config = {}

    # Handle grayscale images
    if image.mode != 'RGB':
        if config.get('force_rgb', True):
            image = image.convert('RGB')
        else:
            # No color correction needed for grayscale
            return image

    # Apply white balance if requested
    if config.get('white_balance', False):
        img_array = np.array(image)
        avg_r = np.mean(img_array[:, :, 0])
        avg_g = np.mean(img_array[:, :, 1])
        avg_b = np.mean(img_array[:, :, 2])
        avg = (avg_r + avg_g + avg_b) / 3

        # Apply correction only if there's a significant color cast
        if max(avg_r, avg_g, avg_b) - min(avg_r, avg_g, avg_b) > 10:
            r_gain = avg / avg_r if avg_r > 0 else 1
            g_gain = avg / avg_g if avg_g > 0 else 1
            b_gain = avg / avg_b if avg_b > 0 else 1

            # Apply gains with clipping
            img_array[:, :, 0] = np.clip(
                img_array[:, :, 0] * r_gain, 0, 255).astype(np.uint8)
            img_array[:, :, 1] = np.clip(
                img_array[:, :, 1] * g_gain, 0, 255).astype(np.uint8)
            img_array[:, :, 2] = np.clip(
                img_array[:, :, 2] * b_gain, 0, 255).astype(np.uint8)

            image = Image.fromarray(img_array)

    # Apply saturation adjustment
    saturation_factor = config.get('saturation_factor')
    if saturation_factor is not None and saturation_factor != 1.0:
        image = ImageEnhance.Color(image).enhance(saturation_factor)

    # Apply color temperature adjustment (warm/cool)
    temperature = config.get('temperature', 0)
    if temperature != 0:
        img_array = np.array(image)
        if temperature > 0:  # Warm (more red, less blue)
            red_factor = 1 + (0.1 * temperature)
            blue_factor = 1 - (0.1 * temperature)
        else:  # Cool (more blue, less red)
            temperature = abs(temperature)
            red_factor = 1 - (0.1 * temperature)
            blue_factor = 1 + (0.1 * temperature)

        # Apply temperature adjustment with clipping
        img_array[:, :, 0] = np.clip(
            img_array[:, :, 0] * red_factor, 0, 255).astype(np.uint8)  # Red
        img_array[:, :, 2] = np.clip(
            img_array[:, :, 2] * blue_factor, 0, 255).astype(np.uint8)  # Blue

        image = Image.fromarray(img_array)

    return image


def remove_bad_quality_image(image: Image.Image, config: dict = None) -> bool:
    """
    Assess image quality and decide if an image should be filtered out.

    Args:
        image: PIL image
        config: Dictionary with quality assessment settings
            - min_std_dev: float, minimum standard deviation for contrast
            - min_brightness: float, minimum average brightness (0-255)
            - max_brightness: float, maximum average brightness (0-255)
            - min_size: tuple, minimum (width, height)

    Returns:
        True if image should be removed, False otherwise
    """
    if config is None:
        config = {}

    # Convert to numpy for analysis
    img_array = np.array(image)

    # Check image dimensions
    min_size = config.get('min_size', (32, 32))
    if image.width < min_size[0] or image.height < min_size[1]:
        return True  # Image too small

    # For grayscale
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:
        # Check contrast (standard deviation)
        std_dev = np.std(img_array)
        min_std = config.get('min_std_dev', 15)
        if std_dev < min_std:
            return True  # Low contrast

        # Check brightness
        avg_brightness = np.mean(img_array)
        min_brightness = config.get('min_brightness', 20)
        max_brightness = config.get('max_brightness', 235)
        if avg_brightness < min_brightness or avg_brightness > max_brightness:
            return True  # Too dark or too bright

    # For color images
    elif len(img_array.shape) == 3:
        # Check contrast in luminance
        gray = cv2.cvtColor(
            img_array, cv2.COLOR_RGB2GRAY) if image.mode == 'RGB' else img_array
        std_dev = np.std(gray)
        min_std = config.get('min_std_dev', 15)
        if std_dev < min_std:
            return True  # Low contrast

        # Check brightness
        avg_brightness = np.mean(gray)
        min_brightness = config.get('min_brightness', 20)
        max_brightness = config.get('max_brightness', 235)
        if avg_brightness < min_brightness or avg_brightness > max_brightness:
            return True  # Too dark or too bright

    return False  # Image passes quality checks


def smart_augmentation(image: Image.Image, severity: str = 'medium') -> list:
    """
    Apply smart augmentation based on image analysis.

    Args:
        image: PIL image
        severity: 'low', 'medium', or 'high' to control augmentation intensity

    Returns:
        List of augmented images
    """
    augmented = []
    img_array = np.array(image)

    # Analyze image to determine best augmentations
    is_dark = np.mean(img_array) < 80
    is_low_contrast = np.std(img_array) < 40

    # Base severity multipliers
    severity_multiplier = {
        'low': 0.5,
        'medium': 1.0,
        'high': 1.5
    }.get(severity, 1.0)

    # Original image
    augmented.append(image)

    # Horizontal flip (always include)
    augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))

    # Random rotation (small angles)
    angle = np.random.uniform(-10, 10) * severity_multiplier
    augmented.append(image.rotate(angle, resample=Image.BICUBIC, expand=False))

    # Brightness adjustment (conditional)
    if is_dark:
        brightness_factor = 1.0 + (0.3 * severity_multiplier)
        augmented.append(ImageEnhance.Brightness(
            image).enhance(brightness_factor))
    else:
        brightness_factors = [0.9, 1.1] if severity == 'low' else [0.8, 1.2]
        for factor in brightness_factors:
            augmented.append(ImageEnhance.Brightness(image).enhance(factor))

    # Contrast adjustment (conditional)
    if is_low_contrast:
        contrast_factor = 1.0 + (0.4 * severity_multiplier)
        augmented.append(ImageEnhance.Contrast(image).enhance(contrast_factor))
    else:
        contrast_factor = 0.9 if severity == 'low' else 0.8
        augmented.append(ImageEnhance.Contrast(image).enhance(contrast_factor))

    # Small affine transformations for 'medium' and 'high' severity
    if severity in ['medium', 'high']:
        # Shear
        shear_angle = np.random.uniform(-5, 5) * severity_multiplier
        augmented.append(image.transform(
            image.size,
            Image.AFFINE,
            (1, shear_angle/100, 0, 0, 1, 0),
            resample=Image.BICUBIC
        ))

        # Slight zoom
        zoom = 1.0 - (0.1 * severity_multiplier)
        w, h = image.size
        zoom_w, zoom_h = int(w * zoom), int(h * zoom)
        left = (w - zoom_w) // 2
        top = (h - zoom_h) // 2
        right = left + zoom_w
        bottom = top + zoom_h
        zoomed = image.crop((left, top, right, bottom)
                            ).resize((w, h), Image.BICUBIC)
        augmented.append(zoomed)

    return augmented


def get_dataloader(
    dataset_dir: str,
    dataset_name: str,
    mode: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    **dl_kwargs
):
    """
    Create a DataLoader with optional class balancing based on config:
    - weighted_sampler: uses get_WeightedRandom_Sampler
    """
    # load dataset
    dataset = preprocess_dataset(dataset_dir, dataset_name, mode)
    cfg = load_preprocessing_config()
    ds_cfg = cfg.get(dataset_name, {})

    # if user wants sampler-based balancing
    if ds_cfg.get("class_balancing") == "weighted_sampler":
        sampler = get_WeightedRandom_Sampler(dataset, dataset)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, **dl_kwargs)

    # default loader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **dl_kwargs)
