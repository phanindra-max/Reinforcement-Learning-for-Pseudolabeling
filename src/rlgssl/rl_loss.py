"""
Reinforcement Learning Loss Function for RLGSSL
Implements KL-divergence weighted negative reward loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from .reward_function import RLRewardFunction, AdaptiveRewardFunction
from .mixup import MixupGenerator


class RLLossFunction:
    """
    RL Loss Function for RLGSSL
    L_rl = -E[y_i^u ~ π_θ] KL(e, y_i^u) * R(s,a;sg[θ])
    where e is uniform distribution and R is the reward function
    """
    
    def __init__(
        self,
        reward_function: RLRewardFunction,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        uniform_baseline: bool = True
    ):
        """
        Initialize RL loss function
        
        Args:
            reward_function: Reward function to use
            device: Device to perform computations on
            uniform_baseline: Whether to use uniform distribution as baseline
        """
        self.reward_function = reward_function
        self.device = device
        self.uniform_baseline = uniform_baseline
    
    def create_uniform_baseline(self, batch_size: int, num_classes: int) -> torch.Tensor:
        """
        Create uniform distribution baseline (e in the paper)
        
        Args:
            batch_size: Number of samples
            num_classes: Number of classes
        
        Returns:
            Uniform distribution tensor [batch_size, num_classes]
        """
        uniform_dist = torch.ones(batch_size, num_classes, device=self.device) / num_classes
        return uniform_dist
    
    def compute_kl_weight(
        self,
        predictions: torch.Tensor,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute KL divergence weight KL(e, y_i^u)
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            baseline: Baseline distribution (uniform if None)
        
        Returns:
            KL divergence weights [batch_size]
        """
        batch_size, num_classes = predictions.shape
        
        if baseline is None:
            baseline = self.create_uniform_baseline(batch_size, num_classes)
        
        # Compute KL divergence: KL(baseline || predictions)
        # KL(P||Q) = Σ P(x) log(P(x)/Q(x))
        log_predictions = torch.log(predictions + 1e-8)
        log_baseline = torch.log(baseline + 1e-8)
        
        kl_div = torch.sum(baseline * (log_baseline - log_predictions), dim=1)
        
        return kl_div
    
    def compute_rl_loss(
        self,
        model: nn.Module,
        unlabeled_data: torch.Tensor,
        mixup_data: torch.Tensor,
        mixup_labels: torch.Tensor,
        reduction: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute RL loss
        L_rl = -E[y_i^u ~ π_θ] KL(e, y_i^u) * R(s,a;sg[θ])
        
        Args:
            model: Policy network (student model)
            unlabeled_data: Unlabeled data used for pseudo-labeling [N_u, C, H, W]
            mixup_data: Mixup data [N_m, C, H, W]
            mixup_labels: Mixup labels [N_m, num_classes]
            reduction: How to reduce the loss ('mean', 'sum', 'none')
        
        Returns:
            Dictionary containing loss components
        """
        # Get pseudo-labels (actions) from unlabeled data
        pseudo_labels = model.get_probabilities(unlabeled_data)  # π_θ(a|s)
        
        # Compute reward on mixup data (with stop gradient)
        reward = self.reward_function.compute_reward(
            model, mixup_data, mixup_labels, stop_gradient=True
        )
        
        # Compute KL divergence weights
        kl_weights = self.compute_kl_weight(pseudo_labels)
        
        # RL loss: -E[KL(e, y_i^u) * R(s,a;sg[θ])]
        # Note: We compute per-sample loss and then reduce
        if reduction == 'mean':
            weighted_loss = -kl_weights.mean() * reward
        elif reduction == 'sum':
            weighted_loss = -kl_weights.sum() * reward
        elif reduction == 'none':
            weighted_loss = -kl_weights * reward
        else:
            raise ValueError(f"Invalid reduction: {reduction}")
        
        return {
            'rl_loss': weighted_loss,
            'reward': reward,
            'kl_weights': kl_weights,
            'kl_weight_mean': kl_weights.mean(),
            'pseudo_labels': pseudo_labels,
            'pseudo_label_entropy': self._compute_entropy(pseudo_labels)
        }
    
    def compute_detailed_rl_loss(
        self,
        model: nn.Module,
        unlabeled_data: torch.Tensor,
        mixup_data: torch.Tensor,
        mixup_labels: torch.Tensor,
        mu_values: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detailed RL loss with additional analysis
        
        Args:
            model: Policy network
            unlabeled_data: Unlabeled data
            mixup_data: Mixup data
            mixup_labels: Mixup labels
            mu_values: Optional mixup ratios
        
        Returns:
            Detailed loss information
        """
        # Basic RL loss
        basic_loss = self.compute_rl_loss(model, unlabeled_data, mixup_data, mixup_labels)
        
        # Detailed reward analysis
        detailed_reward = self.reward_function.compute_detailed_reward(
            model, mixup_data, mixup_labels
        )
        
        # Additional analysis
        pseudo_labels = basic_loss['pseudo_labels']
        
        # Prediction confidence and diversity
        max_probs = torch.max(pseudo_labels, dim=1)[0]
        prediction_confidence = max_probs.mean()
        prediction_diversity = torch.std(max_probs)
        
        # Uniformity measure (how close to uniform are the predictions)
        num_classes = pseudo_labels.size(1)
        batch_mean_pred = pseudo_labels.mean(dim=0)
        uniformity_score = F.kl_div(
            torch.log(batch_mean_pred + 1e-8),
            torch.ones_like(batch_mean_pred) / num_classes,
            reduction='sum'
        )
        
        return {
            **basic_loss,
            **detailed_reward,
            'prediction_confidence': prediction_confidence,
            'prediction_diversity': prediction_diversity,
            'uniformity_score': uniformity_score,
            'batch_mean_prediction': batch_mean_pred
        }
    
    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distributions"""
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()


class AdaptiveRLLossFunction(RLLossFunction):
    """
    Adaptive RL loss function with additional enhancements
    """
    
    def __init__(
        self,
        reward_function: Union[RLRewardFunction, AdaptiveRewardFunction],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        confidence_threshold: float = 0.8,
        entropy_regularization: float = 0.01
    ):
        super().__init__(reward_function, device)
        self.confidence_threshold = confidence_threshold
        self.entropy_regularization = entropy_regularization
    
    def compute_adaptive_rl_loss(
        self,
        model: nn.Module,
        unlabeled_data: torch.Tensor,
        mixup_data: torch.Tensor,
        mixup_labels: torch.Tensor,
        mu_values: torch.Tensor,
        epoch: int = 0,
        max_epochs: int = 400
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adaptive RL loss with curriculum learning
        
        Args:
            model: Policy network
            unlabeled_data: Unlabeled data
            mixup_data: Mixup data
            mixup_labels: Mixup labels
            mu_values: Mixup ratios
            epoch: Current epoch
            max_epochs: Maximum epochs for curriculum
        
        Returns:
            Adaptive loss information
        """
        # Get pseudo-labels
        pseudo_labels = model.get_probabilities(unlabeled_data)
        
        # Compute adaptive reward if available
        if isinstance(self.reward_function, AdaptiveRewardFunction):
            reward_info = self.reward_function.compute_adaptive_reward(
                model, mixup_data, mixup_labels, mu_values
            )
            reward = reward_info['adaptive_reward']
        else:
            reward = self.reward_function.compute_reward(
                model, mixup_data, mixup_labels, stop_gradient=True
            )
            reward_info = {'base_reward': reward}
        
        # Adaptive KL weighting based on confidence
        kl_weights = self.compute_kl_weight(pseudo_labels)
        
        # Confidence-based weighting
        max_probs = torch.max(pseudo_labels, dim=1)[0]
        confidence_mask = (max_probs > self.confidence_threshold).float()
        
        # Curriculum learning: gradually increase focus on high-confidence samples
        curriculum_weight = min(1.0, epoch / (max_epochs * 0.5))  # Ramp up over first half
        adaptive_weights = kl_weights * (1 + curriculum_weight * confidence_mask)
        
        # Entropy regularization to encourage exploration
        entropy_reg = self._compute_entropy(pseudo_labels)
        
        # Adaptive RL loss
        weighted_reward = adaptive_weights.mean() * reward
        rl_loss = -weighted_reward + self.entropy_regularization * entropy_reg
        
        return {
            'adaptive_rl_loss': rl_loss,
            'weighted_reward': weighted_reward,
            'entropy_regularization': entropy_reg,
            'curriculum_weight': curriculum_weight,
            'confidence_ratio': confidence_mask.mean(),
            'adaptive_weights': adaptive_weights,
            **reward_info
        }


class CombinedLossFunction:
    """
    Combined loss function that integrates RL loss with supervised and consistency losses
    L(θ_S) = L_rl + λ₁ * L_sup + λ₂ * L_cons
    """
    
    def __init__(
        self,
        rl_loss_fn: RLLossFunction,
        lambda_sup: float = 0.1,
        lambda_cons: float = 0.1,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize combined loss function
        
        Args:
            rl_loss_fn: RL loss function
            lambda_sup: Weight for supervised loss
            lambda_cons: Weight for consistency loss
            device: Device for computations
        """
        self.rl_loss_fn = rl_loss_fn
        self.lambda_sup = lambda_sup
        self.lambda_cons = lambda_cons
        self.device = device
    
    def compute_combined_loss(
        self,
        framework,  # TeacherStudentFramework
        labeled_data: torch.Tensor,
        labels: torch.Tensor,
        unlabeled_data: torch.Tensor,
        mixup_data: torch.Tensor,
        mixup_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss function
        
        Args:
            framework: Teacher-student framework
            labeled_data: Labeled input data
            labels: True labels
            unlabeled_data: Unlabeled input data
            mixup_data: Mixup data
            mixup_labels: Mixup labels
        
        Returns:
            Combined loss information
        """
        # Compute individual loss components
        rl_loss_info = self.rl_loss_fn.compute_rl_loss(
            framework.student_model, unlabeled_data, mixup_data, mixup_labels
        )
        
        sup_loss = framework.compute_supervised_loss(labeled_data, labels)
        cons_loss = framework.compute_consistency_loss(unlabeled_data)
        
        # Combined loss
        total_loss = (
            rl_loss_info['rl_loss'] +
            self.lambda_sup * sup_loss +
            self.lambda_cons * cons_loss
        )
        
        return {
            'total_loss': total_loss,
            'rl_loss': rl_loss_info['rl_loss'],
            'supervised_loss': sup_loss,
            'consistency_loss': cons_loss,
            'reward': rl_loss_info['reward'],
            'kl_weight_mean': rl_loss_info['kl_weight_mean'],
            'lambda_sup': self.lambda_sup,
            'lambda_cons': self.lambda_cons
        }


def test_rl_loss_function():
    """Test the RL loss function implementation"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test components
    from ..models.networks import create_model
    from .reward_function import RLRewardFunction
    
    model = create_model('cnn13', num_classes=10).to(device)
    reward_fn = RLRewardFunction(device=device)
    rl_loss_fn = RLLossFunction(reward_fn, device=device)
    
    # Test data
    batch_size = 16
    unlabeled_data = torch.randn(batch_size, 3, 32, 32, device=device)
    mixup_data = torch.randn(batch_size, 3, 32, 32, device=device)
    mixup_labels = torch.softmax(torch.randn(batch_size, 10, device=device), dim=1)
    
    print("Testing RL loss function...")
    
    # Test basic RL loss
    loss_info = rl_loss_fn.compute_rl_loss(model, unlabeled_data, mixup_data, mixup_labels)
    print(f"RL Loss: {loss_info['rl_loss'].item():.6f}")
    print(f"Reward: {loss_info['reward'].item():.6f}")
    print(f"KL Weight Mean: {loss_info['kl_weight_mean'].item():.6f}")
    
    # Test detailed loss
    detailed_loss = rl_loss_fn.compute_detailed_rl_loss(
        model, unlabeled_data, mixup_data, mixup_labels
    )
    print(f"Prediction Confidence: {detailed_loss['prediction_confidence'].item():.4f}")
    print(f"Prediction Diversity: {detailed_loss['prediction_diversity'].item():.4f}")
    
    # Test adaptive RL loss
    adaptive_reward_fn = AdaptiveRewardFunction(device=device)
    adaptive_rl_loss_fn = AdaptiveRLLossFunction(adaptive_reward_fn, device=device)
    
    mu_values = torch.rand(batch_size, device=device)
    adaptive_loss = adaptive_rl_loss_fn.compute_adaptive_rl_loss(
        model, unlabeled_data, mixup_data, mixup_labels, mu_values, epoch=50, max_epochs=400
    )
    print(f"Adaptive RL Loss: {adaptive_loss['adaptive_rl_loss'].item():.6f}")
    print(f"Confidence Ratio: {adaptive_loss['confidence_ratio'].item():.4f}")
    
    # Test gradient flow
    print(f"\nTesting gradient flow...")
    model.zero_grad()
    loss_info['rl_loss'].backward()
    
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradients computed: {has_gradients}")
    
    if has_gradients:
        total_grad_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach()) for p in model.parameters() 
            if p.grad is not None
        ]))
        print(f"Total gradient norm: {total_grad_norm.item():.6f}")
    
    print("RL loss function test completed!")


if __name__ == "__main__":
    test_rl_loss_function()
