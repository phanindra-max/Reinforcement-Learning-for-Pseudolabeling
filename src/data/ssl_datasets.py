"""
Semi-Supervised Learning Dataset Implementation for RLGSSL
Handles CIFAR-10, CIFAR-100, and SVHN with limited labeled data scenarios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import random


class SSLDatasetSplitter:
    """
    Splits datasets into labeled and unlabeled portions for semi-supervised learning
    Follows the paper's experimental setup for CIFAR-10/100 and SVHN
    """
    
    def __init__(self, dataset_name: str, num_labeled: int, seed: int = 42):
        self.dataset_name = dataset_name.lower()
        self.num_labeled = num_labeled
        self.seed = seed
        self.num_classes = self._get_num_classes()
        
    def _get_num_classes(self) -> int:
        """Get number of classes for each dataset"""
        if self.dataset_name == 'cifar10' or self.dataset_name == 'svhn':
            return 10
        elif self.dataset_name == 'cifar100':
            return 100
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
    
    def split_dataset(self, train_dataset) -> Tuple[List[int], List[int]]:
        """
        Split dataset into labeled and unlabeled indices
        Ensures balanced labeled set across all classes
        Fast-paths by reading label arrays directly instead of iterating __getitem__
        """
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # Prefer fast access to labels without materializing images
        labels_array = None
        if hasattr(train_dataset, 'targets'):
            labels_array = train_dataset.targets
        elif hasattr(train_dataset, 'labels'):
            labels_array = train_dataset.labels
        
        class_indices = {}
        if labels_array is not None:
            # Normalize to plain list for iteration speed
            if isinstance(labels_array, np.ndarray):
                labels_list = labels_array.tolist()
            else:
                labels_list = list(labels_array)
            for idx, label in enumerate(labels_list):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
        else:
            # Fallback: iterate dataset (slower)
            for idx, (_, label) in enumerate(train_dataset):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
        
        # Shuffle indices within each class
        for label in class_indices:
            np.random.shuffle(class_indices[label])
        
        # Select balanced labeled samples
        labeled_per_class = self.num_labeled // self.num_classes
        labeled_indices = []
        
        for label in range(self.num_classes):
            if label in class_indices:
                # Take first `labeled_per_class` samples from each class
                labeled_indices.extend(class_indices[label][:labeled_per_class])
        
        # Handle any remaining samples if num_labeled not perfectly divisible
        remaining = self.num_labeled - len(labeled_indices)
        if remaining > 0:
            all_remaining = []
            for label in range(self.num_classes):
                if label in class_indices:
                    all_remaining.extend(class_indices[label][labeled_per_class:])
            np.random.shuffle(all_remaining)
            labeled_indices.extend(all_remaining[:remaining])
        
        # All other indices are unlabeled
        all_indices = set(range(len(train_dataset)))
        labeled_indices = set(labeled_indices)
        unlabeled_indices = list(all_indices - labeled_indices)
        
        return list(labeled_indices), unlabeled_indices


class SSLTransforms:
    """
    Data augmentation transforms for semi-supervised learning
    Includes weak and strong augmentations as commonly used in SSL
    """
    
    def __init__(self, dataset_name: str, is_training: bool = True):
        self.dataset_name = dataset_name.lower()
        self.is_training = is_training
    
    def get_weak_transform(self):
        """Weak augmentation - basic transforms"""
        if self.dataset_name in ['cifar10', 'cifar100']:
            if self.is_training:
                return transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
        
        elif self.dataset_name == 'svhn':
            if self.is_training:
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                ])
            else:
                return transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                ])
    
    def get_strong_transform(self):
        """Strong augmentation - more aggressive transforms for consistency training"""
        weak = self.get_weak_transform()
        if self.is_training:
            # Add stronger augmentations to the weak transform
            strong_ops = [
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ]
            
            # Insert strong ops before ToTensor
            ops = weak.transforms[:-2] + strong_ops + weak.transforms[-2:]
            return transforms.Compose(ops)
        else:
            return weak


class SSLDataset(Dataset):
    """
    Semi-supervised learning dataset wrapper
    Returns both weak and strong augmentations for consistency training
    """
    
    def __init__(self, dataset, indices: List[int], transform=None, is_labeled: bool = True):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.is_labeled = is_labeled
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.dataset[actual_idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_labeled:
            return image, label, actual_idx
        else:
            # For unlabeled data, we might want multiple augmentations
            return image, label, actual_idx  # label is available but not used in loss


def create_ssl_dataloaders(
    dataset_name: str,
    num_labeled: int,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
    data_root: str = './data',
    prefetch_factor: int = 2,
    persistent_workers: bool = True
) -> Dict[str, DataLoader]:
    """
    Create semi-supervised learning dataloaders
    
    Args:
        dataset_name: 'cifar10', 'cifar100', or 'svhn'
        num_labeled: Number of labeled samples to use
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
        data_root: Root directory for datasets
    
    Returns:
        Dictionary containing train_labeled, train_unlabeled, and test dataloaders
    """
    
    # Initialize transforms
    ssl_transforms = SSLTransforms(dataset_name, is_training=True)
    test_transforms = SSLTransforms(dataset_name, is_training=False)
    
    weak_transform = ssl_transforms.get_weak_transform()
    test_transform = test_transforms.get_weak_transform()
    
    # Load datasets
    if dataset_name.lower() == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=None
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=test_transform
        )
    
    elif dataset_name.lower() == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=None
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=test_transform
        )
    
    elif dataset_name.lower() == 'svhn':
        train_dataset = torchvision.datasets.SVHN(
            root=data_root, split='train', download=True, transform=None
        )
        test_dataset = torchvision.datasets.SVHN(
            root=data_root, split='test', download=True, transform=test_transform
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Split into labeled and unlabeled
    splitter = SSLDatasetSplitter(dataset_name, num_labeled, seed)
    labeled_indices, unlabeled_indices = splitter.split_dataset(train_dataset)
    
    print(f"Dataset: {dataset_name}")
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Labeled samples: {len(labeled_indices)}")
    print(f"Unlabeled samples: {len(unlabeled_indices)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create SSL datasets
    labeled_dataset = SSLDataset(
        train_dataset, labeled_indices, transform=weak_transform, is_labeled=True
    )
    unlabeled_dataset = SSLDataset(
        train_dataset, unlabeled_indices, transform=weak_transform, is_labeled=False
    )
    
    # Common DataLoader kwargs with conditional performance flags
    def loader_kwargs(shuffle: bool, drop_last: bool = False) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
            'drop_last': drop_last,
        }
        if num_workers > 0:
            kwargs['persistent_workers'] = persistent_workers
            kwargs['prefetch_factor'] = prefetch_factor
        return kwargs

    # Create dataloaders
    train_labeled_loader = DataLoader(
        labeled_dataset,
        **loader_kwargs(shuffle=True, drop_last=True)
    )
    
    train_unlabeled_loader = DataLoader(
        unlabeled_dataset,
        **loader_kwargs(shuffle=True, drop_last=True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        **loader_kwargs(shuffle=False, drop_last=False)
    )
    
    return {
        'train_labeled': train_labeled_loader,
        'train_unlabeled': train_unlabeled_loader,
        'test': test_loader,
        'num_classes': splitter.num_classes
    }


# Example usage and testing
if __name__ == "__main__":
    # Test the data loading
    dataloaders = create_ssl_dataloaders(
        dataset_name='cifar10',
        num_labeled=1000,
        batch_size=64,
        seed=42
    )
    
    print("Testing labeled dataloader...")
    for batch_idx, (data, target, indices) in enumerate(dataloaders['train_labeled']):
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        if batch_idx >= 2:  # Just test a few batches
            break
    
    print("\nTesting unlabeled dataloader...")
    for batch_idx, (data, target, indices) in enumerate(dataloaders['train_unlabeled']):
        print(f"Batch {batch_idx}: Data shape: {data.shape}, Target shape: {target.shape}")
        if batch_idx >= 2:  # Just test a few batches
            break
