"""
Mixup Implementation for RLGSSL Reward Function
Handles linear interpolation between labeled and pseudo-labeled data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random


class MixupGenerator:
    """
    Generates mixup data for RLGSSL reward function
    Handles size mismatch between labeled and unlabeled data via replication
    """
    
    def __init__(self, alpha: float = 1.0, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize mixup generator
        
        Args:
            alpha: Beta distribution parameter for mixup ratio (λ ~ Beta(α, α))
            device: Device to perform operations on
        """
        self.alpha = alpha
        self.device = device
    
    def sample_mixup_ratio(self, batch_size: int) -> torch.Tensor:
        """
        Sample mixup ratio λ from Beta distribution
        
        Args:
            batch_size: Number of samples to generate ratios for
        
        Returns:
            Mixup ratios tensor of shape (batch_size,)
        """
        if self.alpha > 0:
            # Sample from Beta(α, α) distribution
            ratios = np.random.beta(self.alpha, self.alpha, batch_size)
        else:
            # Fixed ratio of 0.5 if alpha = 0
            ratios = np.full(batch_size, 0.5)
        
        return torch.tensor(ratios, dtype=torch.float32, device=self.device)
    
    def replicate_labeled_data(
        self, 
        labeled_data: torch.Tensor, 
        labeled_labels: torch.Tensor,
        target_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Replicate labeled data to match unlabeled data size
        
        Args:
            labeled_data: Labeled input data [N_l, C, H, W]
            labeled_labels: Labeled targets [N_l, num_classes] (one-hot or probabilities)
            target_size: Target size (N_u)
        
        Returns:
            Replicated data and labels
        """
        n_labeled = labeled_data.size(0)
        
        if target_size <= n_labeled:
            # If target size is smaller, just sample
            indices = torch.randperm(n_labeled, device=self.device)[:target_size]
            return labeled_data[indices], labeled_labels[indices]
        
        # Calculate replication factor
        replication_factor = (target_size + n_labeled - 1) // n_labeled  # Ceiling division
        
        # Replicate data
        replicated_data = labeled_data.repeat(replication_factor, 1, 1, 1)
        replicated_labels = labeled_labels.repeat(replication_factor, 1)
        
        # Trim to exact target size and shuffle
        indices = torch.randperm(replicated_data.size(0), device=self.device)[:target_size]
        
        return replicated_data[indices], replicated_labels[indices]
    
    def create_mixup_data(
        self,
        labeled_data: torch.Tensor,
        labeled_labels: torch.Tensor,
        unlabeled_data: torch.Tensor,
        pseudo_labels: torch.Tensor,
        mu_values: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create mixup data between labeled and pseudo-labeled samples
        
        Formula:
        x_i^m = μ * x_i^u + (1-μ) * x_i^l
        y_i^m = μ * y_i^u + (1-μ) * y_i^l
        
        Args:
            labeled_data: Labeled input data [N_l, C, H, W]
            labeled_labels: Labeled targets [N_l, num_classes] (one-hot or probabilities)
            unlabeled_data: Unlabeled input data [N_u, C, H, W]
            pseudo_labels: Pseudo-labels [N_u, num_classes] (probabilities)
            mu_values: Optional pre-computed mixup ratios [N_u]
        
        Returns:
            Tuple of (mixup_data, mixup_labels, mu_values)
        """
        n_unlabeled = unlabeled_data.size(0)
        n_labeled = labeled_data.size(0)
        
        # Handle size mismatch by replicating labeled data
        if n_unlabeled != n_labeled:
            labeled_data_rep, labeled_labels_rep = self.replicate_labeled_data(
                labeled_data, labeled_labels, n_unlabeled
            )
        else:
            labeled_data_rep, labeled_labels_rep = labeled_data, labeled_labels
        
        # Sample mixup ratios if not provided
        if mu_values is None:
            mu_values = self.sample_mixup_ratio(n_unlabeled)
        
        # Reshape mu for broadcasting: [N_u] -> [N_u, 1, 1, 1] for data, [N_u, 1] for labels
        mu_data = mu_values.view(-1, 1, 1, 1)
        mu_labels = mu_values.view(-1, 1)
        
        # Create mixup data and labels
        mixup_data = mu_data * unlabeled_data + (1 - mu_data) * labeled_data_rep
        mixup_labels = mu_labels * pseudo_labels + (1 - mu_labels) * labeled_labels_rep
        
        return mixup_data, mixup_labels, mu_values
    
    def create_balanced_mixup_batch(
        self,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Dict[str, torch.Tensor],
        pseudo_labels: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Create a balanced mixup batch for training
        
        Args:
            labeled_batch: Dictionary with 'data' and 'labels' keys
            unlabeled_batch: Dictionary with 'data' key
            pseudo_labels: Pseudo-labels for unlabeled data [N_u, num_classes]
            batch_size: Optional batch size for output (defaults to unlabeled batch size)
        
        Returns:
            Dictionary with mixup data, labels, and metadata
        """
        labeled_data = labeled_batch['data']
        labeled_labels = labeled_batch['labels']
        unlabeled_data = unlabeled_batch['data']
        
        # Convert hard labels to soft labels if needed
        if len(labeled_labels.shape) == 1:
            # Convert to one-hot
            num_classes = pseudo_labels.size(1)
            labeled_labels = F.one_hot(labeled_labels, num_classes).float()
        
        # Create mixup data
        mixup_data, mixup_labels, mu_values = self.create_mixup_data(
            labeled_data, labeled_labels, unlabeled_data, pseudo_labels
        )
        
        # Optionally subsample if batch_size is specified
        if batch_size is not None and mixup_data.size(0) > batch_size:
            indices = torch.randperm(mixup_data.size(0), device=self.device)[:batch_size]
            mixup_data = mixup_data[indices]
            mixup_labels = mixup_labels[indices]
            mu_values = mu_values[indices]
        
        return {
            'data': mixup_data,
            'labels': mixup_labels,
            'mu_values': mu_values,
            'metadata': {
                'original_labeled_size': labeled_data.size(0),
                'original_unlabeled_size': unlabeled_data.size(0),
                'mixup_size': mixup_data.size(0)
            }
        }


class AdaptiveMixupGenerator(MixupGenerator):
    """
    Adaptive mixup generator that adjusts mixing strategy based on training progress
    """
    
    def __init__(
        self, 
        alpha: float = 1.0, 
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        adaptive_strategy: str = 'confidence_based'
    ):
        super().__init__(alpha, device)
        self.adaptive_strategy = adaptive_strategy
        self.confidence_history = []
    
    def adaptive_mixup_ratio(
        self,
        pseudo_labels: torch.Tensor,
        confidence_threshold: float = 0.8
    ) -> torch.Tensor:
        """
        Generate adaptive mixup ratios based on pseudo-label confidence
        
        Args:
            pseudo_labels: Pseudo-labels [N, num_classes]
            confidence_threshold: Threshold for high-confidence samples
        
        Returns:
            Adaptive mixup ratios
        """
        batch_size = pseudo_labels.size(0)
        
        if self.adaptive_strategy == 'confidence_based':
            # Higher mu (more unlabeled) for high-confidence pseudo-labels
            max_probs = torch.max(pseudo_labels, dim=1)[0]
            confidence_scores = max_probs
            
            # Scale mu based on confidence: high confidence -> higher mu
            base_mu = self.sample_mixup_ratio(batch_size)
            confidence_boost = (confidence_scores > confidence_threshold).float() * 0.2
            
            adaptive_mu = torch.clamp(base_mu + confidence_boost, 0.0, 1.0)
            return adaptive_mu
        
        elif self.adaptive_strategy == 'entropy_based':
            # Lower mu (more labeled) for high-entropy (uncertain) pseudo-labels
            entropy = -torch.sum(pseudo_labels * torch.log(pseudo_labels + 1e-8), dim=1)
            normalized_entropy = entropy / torch.log(torch.tensor(pseudo_labels.size(1), dtype=torch.float32))
            
            base_mu = self.sample_mixup_ratio(batch_size)
            entropy_penalty = normalized_entropy * 0.3
            
            adaptive_mu = torch.clamp(base_mu - entropy_penalty, 0.0, 1.0)
            return adaptive_mu
        
        else:
            # Default to standard sampling
            return self.sample_mixup_ratio(batch_size)
    
    def create_adaptive_mixup_data(
        self,
        labeled_data: torch.Tensor,
        labeled_labels: torch.Tensor,
        unlabeled_data: torch.Tensor,
        pseudo_labels: torch.Tensor,
        confidence_threshold: float = 0.8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create mixup data with adaptive mixing ratios
        """
        mu_values = self.adaptive_mixup_ratio(pseudo_labels, confidence_threshold)
        
        return self.create_mixup_data(
            labeled_data, labeled_labels, unlabeled_data, pseudo_labels, mu_values
        )


def test_mixup_functionality():
    """Test function for mixup functionality"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    n_labeled, n_unlabeled = 32, 128
    num_classes = 10
    
    labeled_data = torch.randn(n_labeled, 3, 32, 32, device=device)
    labeled_labels = torch.randint(0, num_classes, (n_labeled,), device=device)
    unlabeled_data = torch.randn(n_unlabeled, 3, 32, 32, device=device)
    pseudo_labels = torch.softmax(torch.randn(n_unlabeled, num_classes, device=device), dim=1)
    
    print(f"Test data shapes:")
    print(f"Labeled data: {labeled_data.shape}")
    print(f"Labeled labels: {labeled_labels.shape}")
    print(f"Unlabeled data: {unlabeled_data.shape}")
    print(f"Pseudo labels: {pseudo_labels.shape}")
    
    # Test standard mixup
    mixup_gen = MixupGenerator(alpha=1.0, device=device)
    
    labeled_batch = {
        'data': labeled_data,
        'labels': labeled_labels
    }
    unlabeled_batch = {
        'data': unlabeled_data
    }
    
    mixup_result = mixup_gen.create_balanced_mixup_batch(
        labeled_batch, unlabeled_batch, pseudo_labels
    )
    
    print(f"\nMixup results:")
    print(f"Mixup data shape: {mixup_result['data'].shape}")
    print(f"Mixup labels shape: {mixup_result['labels'].shape}")
    print(f"Mu values shape: {mixup_result['mu_values'].shape}")
    print(f"Mu values range: [{mixup_result['mu_values'].min():.3f}, {mixup_result['mu_values'].max():.3f}]")
    
    # Test adaptive mixup
    adaptive_gen = AdaptiveMixupGenerator(alpha=1.0, device=device, adaptive_strategy='confidence_based')
    
    adaptive_mixup_data, adaptive_mixup_labels, adaptive_mu = adaptive_gen.create_adaptive_mixup_data(
        labeled_data, F.one_hot(labeled_labels, num_classes).float(),
        unlabeled_data, pseudo_labels
    )
    
    print(f"\nAdaptive mixup results:")
    print(f"Adaptive mixup data shape: {adaptive_mixup_data.shape}")
    print(f"Adaptive mixup labels shape: {adaptive_mixup_labels.shape}")
    print(f"Adaptive mu values range: [{adaptive_mu.min():.3f}, {adaptive_mu.max():.3f}]")
    
    # Verify mixup properties
    sample_idx = 0
    mu_val = mixup_result['mu_values'][sample_idx].item()
    
    # Check if mixup formula is correctly applied
    expected_data = mu_val * unlabeled_data[sample_idx] + (1 - mu_val) * labeled_data[0]  # First labeled sample
    actual_data = mixup_result['data'][sample_idx]
    
    data_diff = torch.norm(expected_data - actual_data).item()
    print(f"\nMixup verification (sample {sample_idx}):")
    print(f"Mu value: {mu_val:.3f}")
    print(f"Data difference: {data_diff:.6f} (should be close to 0)")
    
    print("Mixup functionality test completed!")


if __name__ == "__main__":
    test_mixup_functionality()
