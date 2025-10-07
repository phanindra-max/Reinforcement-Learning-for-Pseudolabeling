"""
Reinforcement Learning Reward Function for RLGSSL
Implements the prediction assessment reward based on mixup data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import numpy as np


class RLRewardFunction:
    """
    Reward function for RLGSSL based on mixup data prediction accuracy
    R(s,a;sg[θ]) = -1/(C·N^m) * Σ ||P_θ(x_i^m) - y_i^m||_2^2
    """
    
    def __init__(self, device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize reward function
        
        Args:
            device: Device to perform computations on
        """
        self.device = device
    
    def compute_reward(
        self,
        model: nn.Module,
        mixup_data: torch.Tensor,
        mixup_labels: torch.Tensor,
        stop_gradient: bool = True
    ) -> torch.Tensor:
        """
        Compute reward based on model predictions on mixup data
        
        Args:
            model: The model to evaluate (policy network)
            mixup_data: Mixed data [N^m, C, H, W]
            mixup_labels: Mixed labels [N^m, num_classes]
            stop_gradient: Whether to stop gradients through the model (sg[θ])
        
        Returns:
            Reward value (scalar tensor)
        """
        # Get model predictions on mixup data
        if stop_gradient:
            with torch.no_grad():
                model_predictions = model.get_probabilities(mixup_data)
        else:
            model_predictions = model.get_probabilities(mixup_data)
        
        # Compute negative MSE as reward
        mse = F.mse_loss(model_predictions, mixup_labels, reduction='mean')
        reward = -mse
        
        return reward
    
    def compute_detailed_reward(
        self,
        model: nn.Module,
        mixup_data: torch.Tensor,
        mixup_labels: torch.Tensor,
        stop_gradient: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detailed reward information for analysis
        
        Args:
            model: The model to evaluate
            mixup_data: Mixed data [N^m, C, H, W]
            mixup_labels: Mixed labels [N^m, num_classes]
            stop_gradient: Whether to stop gradients through the model
        
        Returns:
            Dictionary with detailed reward components
        """
        # Get model predictions
        if stop_gradient:
            with torch.no_grad():
                model_predictions = model.get_probabilities(mixup_data)
        else:
            model_predictions = model.get_probabilities(mixup_data)
        
        # Compute various metrics
        mse = F.mse_loss(model_predictions, mixup_labels, reduction='mean')
        per_sample_mse = F.mse_loss(model_predictions, mixup_labels, reduction='none').mean(dim=1)
        
        # L1 loss (alternative metric)
        l1_loss = F.l1_loss(model_predictions, mixup_labels, reduction='mean')
        
        # KL divergence (if mixup_labels are probabilities)
        kl_div = F.kl_div(
            F.log_softmax(model.get_logits(mixup_data), dim=1),
            mixup_labels,
            reduction='batchmean'
        )
        
        # Accuracy-like metric (prediction confidence)
        pred_confidence = torch.max(model_predictions, dim=1)[0].mean()
        label_confidence = torch.max(mixup_labels, dim=1)[0].mean()
        
        return {
            'reward': -mse,
            'mse': mse,
            'per_sample_mse': per_sample_mse,
            'l1_loss': l1_loss,
            'kl_divergence': kl_div,
            'prediction_confidence': pred_confidence,
            'label_confidence': label_confidence,
            'prediction_entropy': self._compute_entropy(model_predictions),
            'label_entropy': self._compute_entropy(mixup_labels)
        }
    
    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distributions"""
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
    
    def compute_batch_rewards(
        self,
        model: nn.Module,
        mixup_data: torch.Tensor,
        mixup_labels: torch.Tensor,
        batch_size: Optional[int] = None,
        stop_gradient: bool = True
    ) -> torch.Tensor:
        """
        Compute rewards for individual samples in the batch
        
        Args:
            model: The model to evaluate
            mixup_data: Mixed data [N^m, C, H, W]
            mixup_labels: Mixed labels [N^m, num_classes]
            batch_size: Optional batch size for processing
            stop_gradient: Whether to stop gradients
        
        Returns:
            Per-sample rewards [N^m]
        """
        if batch_size is None or mixup_data.size(0) <= batch_size:
            # Process all at once
            if stop_gradient:
                with torch.no_grad():
                    model_predictions = model.get_probabilities(mixup_data)
            else:
                model_predictions = model.get_probabilities(mixup_data)
            
            # Per-sample MSE (negative for reward)
            per_sample_rewards = -F.mse_loss(
                model_predictions, mixup_labels, reduction='none'
            ).mean(dim=1)
            
            return per_sample_rewards
        
        else:
            # Process in batches
            all_rewards = []
            num_samples = mixup_data.size(0)
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_data = mixup_data[i:end_idx]
                batch_labels = mixup_labels[i:end_idx]
                
                if stop_gradient:
                    with torch.no_grad():
                        batch_predictions = model.get_probabilities(batch_data)
                else:
                    batch_predictions = model.get_probabilities(batch_data)
                
                batch_rewards = -F.mse_loss(
                    batch_predictions, batch_labels, reduction='none'
                ).mean(dim=1)
                
                all_rewards.append(batch_rewards)
            
            return torch.cat(all_rewards, dim=0)


class AdaptiveRewardFunction(RLRewardFunction):
    """
    Adaptive reward function that can incorporate additional factors
    """
    
    def __init__(
        self, 
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        confidence_weight: float = 0.1,
        diversity_weight: float = 0.05
    ):
        super().__init__(device)
        self.confidence_weight = confidence_weight
        self.diversity_weight = diversity_weight
    
    def compute_adaptive_reward(
        self,
        model: nn.Module,
        mixup_data: torch.Tensor,
        mixup_labels: torch.Tensor,
        mu_values: torch.Tensor,
        stop_gradient: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive reward incorporating confidence and diversity
        
        Args:
            model: The model to evaluate
            mixup_data: Mixed data [N^m, C, H, W]
            mixup_labels: Mixed labels [N^m, num_classes]
            mu_values: Mixup ratios used [N^m]
            stop_gradient: Whether to stop gradients
        
        Returns:
            Dictionary with reward components
        """
        # Base reward
        base_reward_info = self.compute_detailed_reward(
            model, mixup_data, mixup_labels, stop_gradient
        )
        
        # Additional components
        if stop_gradient:
            with torch.no_grad():
                model_predictions = model.get_probabilities(mixup_data)
        else:
            model_predictions = model.get_probabilities(mixup_data)
        
        # Confidence bonus: reward high-confidence correct predictions
        confidence_bonus = self._compute_confidence_bonus(model_predictions, mixup_labels)
        
        # Diversity bonus: encourage diverse predictions across batch
        diversity_bonus = self._compute_diversity_bonus(model_predictions)
        
        # Mixup-aware bonus: reward based on mixup ratio effectiveness
        mixup_bonus = self._compute_mixup_bonus(model_predictions, mixup_labels, mu_values)
        
        # Combined adaptive reward
        adaptive_reward = (
            base_reward_info['reward'] +
            self.confidence_weight * confidence_bonus +
            self.diversity_weight * diversity_bonus
        )
        
        return {
            'adaptive_reward': adaptive_reward,
            'base_reward': base_reward_info['reward'],
            'confidence_bonus': confidence_bonus,
            'diversity_bonus': diversity_bonus,
            'mixup_bonus': mixup_bonus,
            'detailed_info': base_reward_info
        }
    
    def _compute_confidence_bonus(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute confidence-based bonus"""
        # High confidence when prediction and target align
        pred_confidence = torch.max(predictions, dim=1)[0]
        target_confidence = torch.max(targets, dim=1)[0]
        
        # Bonus for high confidence on high-confidence targets
        confidence_alignment = pred_confidence * target_confidence
        return confidence_alignment.mean()
    
    def _compute_diversity_bonus(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute diversity bonus to encourage exploration"""
        # Encourage diverse predictions across the batch
        batch_mean_pred = predictions.mean(dim=0)
        uniformity = -torch.sum(batch_mean_pred * torch.log(batch_mean_pred + 1e-8))
        
        # Normalize by log(num_classes) to get value between 0 and 1
        max_entropy = torch.log(torch.tensor(predictions.size(1), dtype=torch.float32))
        normalized_uniformity = uniformity / max_entropy
        
        return normalized_uniformity
    
    def _compute_mixup_bonus(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mu_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute bonus based on mixup effectiveness"""
        # Reward when the model performs well on different mixup ratios
        per_sample_accuracy = -F.mse_loss(predictions, targets, reduction='none').mean(dim=1)
        
        # Weight by mu diversity (encourage good performance across different mu values)
        mu_diversity = torch.std(mu_values)
        mixup_effectiveness = per_sample_accuracy.mean() * (1 + mu_diversity)
        
        return mixup_effectiveness


def test_reward_function():
    """Test the reward function implementation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test model and data
    from ..models.networks import create_model
    
    model = create_model('cnn13', num_classes=10).to(device)
    
    # Test data
    batch_size = 16
    mixup_data = torch.randn(batch_size, 3, 32, 32, device=device)
    mixup_labels = torch.softmax(torch.randn(batch_size, 10, device=device), dim=1)
    mu_values = torch.rand(batch_size, device=device)
    
    print("Testing reward function...")
    
    # Test basic reward function
    reward_fn = RLRewardFunction(device=device)
    
    # Test basic reward
    reward = reward_fn.compute_reward(model, mixup_data, mixup_labels)
    print(f"Basic reward: {reward.item():.6f}")
    
    # Test detailed reward
    detailed_reward = reward_fn.compute_detailed_reward(model, mixup_data, mixup_labels)
    print(f"Detailed reward keys: {list(detailed_reward.keys())}")
    print(f"MSE: {detailed_reward['mse'].item():.6f}")
    print(f"KL divergence: {detailed_reward['kl_divergence'].item():.6f}")
    
    # Test batch rewards
    batch_rewards = reward_fn.compute_batch_rewards(model, mixup_data, mixup_labels)
    print(f"Batch rewards shape: {batch_rewards.shape}")
    print(f"Batch rewards range: [{batch_rewards.min().item():.6f}, {batch_rewards.max().item():.6f}]")
    
    # Test adaptive reward function
    adaptive_reward_fn = AdaptiveRewardFunction(device=device)
    adaptive_result = adaptive_reward_fn.compute_adaptive_reward(
        model, mixup_data, mixup_labels, mu_values
    )
    
    print(f"\nAdaptive reward: {adaptive_result['adaptive_reward'].item():.6f}")
    print(f"Base reward: {adaptive_result['base_reward'].item():.6f}")
    print(f"Confidence bonus: {adaptive_result['confidence_bonus'].item():.6f}")
    print(f"Diversity bonus: {adaptive_result['diversity_bonus'].item():.6f}")
    
    # Test gradient flow
    print(f"\nTesting gradient flow...")
    reward_with_grad = reward_fn.compute_reward(model, mixup_data, mixup_labels, stop_gradient=False)
    reward_with_grad.backward()
    
    # Check if gradients are computed
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradients computed: {has_gradients}")
    
    print("Reward function test completed!")


if __name__ == "__main__":
    test_reward_function()
