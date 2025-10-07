"""
Teacher-Student Framework for RLGSSL
Implements EMA-based teacher model for stable pseudo-label generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from tqdm import tqdm
import copy


class EMATeacher:
    """
    Exponential Moving Average Teacher Model
    Maintains a stable version of the student model for pseudo-label generation
    """
    
    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999):
        """
        Initialize EMA teacher
        
        Args:
            student_model: The student model to track
            ema_decay: EMA decay rate (β in the paper)
        """
        self.ema_decay = ema_decay
        self.student_model = student_model
        
        # Create teacher model as a copy of student
        self.teacher_model = copy.deepcopy(student_model)
        
        # Set teacher to eval mode and disable gradients
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def update(self):
        """
        Update teacher model parameters using EMA
        θ_T = β * θ_T + (1 - β) * θ_S
        """
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(), 
                self.student_model.parameters()
            ):
                teacher_param.data.mul_(self.ema_decay).add_(
                    student_param.data, alpha=1 - self.ema_decay
                )
    
    def __call__(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass through teacher model"""
        self.teacher_model.eval()
        with torch.no_grad():
            return self.teacher_model(x, return_logits=return_logits)
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities from teacher model"""
        return self(x, return_logits=False)
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get logits from teacher model"""
        return self(x, return_logits=True)


class TeacherStudentFramework:
    """
    Complete Teacher-Student Framework for RLGSSL
    Manages both student and teacher models with various loss components
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        ema_decay: float = 0.999,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize teacher-student framework
        
        Args:
            student_model: The student model to train
            ema_decay: EMA decay rate for teacher
            device: Device to run models on
        """
        self.student_model = student_model.to(device)
        self.teacher = EMATeacher(student_model, ema_decay)
        self.device = device
        
        # Move teacher to device
        self.teacher.teacher_model = self.teacher.teacher_model.to(device)
    
    def forward_student(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass through student model"""
        return self.student_model(x, return_logits=return_logits)
    
    def forward_teacher(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """Forward pass through teacher model"""
        return self.teacher(x, return_logits=return_logits)
    
    def update_teacher(self):
        """Update teacher model with EMA"""
        self.teacher.update()
    
    def compute_supervised_loss(
        self, 
        labeled_data: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute supervised loss on labeled data
        L_sup = E[(x^l, y^l) ∈ D^l] [l_CE(P_θs(x^l), y^l)]
        
        Args:
            labeled_data: Labeled input data
            labels: True labels
        
        Returns:
            Cross-entropy loss
        """
        student_logits = self.forward_student(labeled_data, return_logits=True)
        return F.cross_entropy(student_logits, labels)
    
    def compute_consistency_loss(
        self, 
        unlabeled_data: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute consistency loss between teacher and student predictions
        L_cons = E[x^u ∈ D^u] [l_KL(P_θs(x^u), P_θT(x^u))]
        
        Args:
            unlabeled_data: Unlabeled input data
            temperature: Temperature for softmax (for sharpening)
        
        Returns:
            KL divergence loss
        """
        # Get teacher predictions (stable, no gradients)
        teacher_probs = self.forward_teacher(unlabeled_data, return_logits=False)
        
        # Get student predictions (with gradients)
        student_logits = self.forward_student(unlabeled_data, return_logits=True)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        
        # KL divergence: KL(teacher || student)
        consistency_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        )
        
        return consistency_loss
    
    def evaluate(self, test_loader) -> Dict[str, float]:
        """
        Evaluate both student and teacher models (robust to 2- or 3-tuple batches)
        """
        self.student_model.eval()
        
        student_correct = 0
        teacher_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        data, targets = batch[0], batch[1]
                    else:
                        continue
                else:
                    # Unsupported format
                    continue
                
                data = data.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Student predictions
                student_logits = self.forward_student(data, return_logits=True)
                student_pred = student_logits.argmax(dim=1)
                student_correct += (student_pred == targets).sum().item()
                
                # Teacher predictions
                teacher_logits = self.forward_teacher(data, return_logits=True)
                teacher_pred = teacher_logits.argmax(dim=1)
                teacher_correct += (teacher_pred == targets).sum().item()
                
                total += targets.size(0)
        
        student_accuracy = 100.0 * student_correct / max(1, total)
        teacher_accuracy = 100.0 * teacher_correct / max(1, total)
        
        return {
            'student_accuracy': student_accuracy,
            'teacher_accuracy': teacher_accuracy,
            'student_error': 100.0 - student_accuracy,
            'teacher_error': 100.0 - teacher_accuracy
        }


def warmup_teacher_student(
    framework: 'TeacherStudentFramework',
    labeled_loader,
    unlabeled_loader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 50,
    lambda_sup: float = 1.0,
    lambda_cons: float = 1.0,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
):
    """
    Warmup training with optional AMP and robust batch unpacking.
    """
    device = framework.device if hasattr(framework, 'device') else next(framework.student_model.parameters()).device
    use_amp = scaler is not None and device.type == 'cuda'
    framework.student_model.train()
    
    for epoch in range(num_epochs):
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        progress = tqdm(total=min(len(labeled_loader), len(unlabeled_loader)), desc=f"Warmup {epoch+1}/{num_epochs}")
        
        while True:
            try:
                batch_l = next(labeled_iter)
                batch_u = next(unlabeled_iter)
            except StopIteration:
                break
            
            # Unpack batches (support 2- or 3-tuples)
            if isinstance(batch_l, (list, tuple)) and len(batch_l) >= 2:
                x_l, y_l = batch_l[0], batch_l[1]
            else:
                continue
            x_u = batch_u[0] if isinstance(batch_u, (list, tuple)) else batch_u
            
            x_l = x_l.to(device, non_blocking=True)
            y_l = y_l.to(device, non_blocking=True)
            x_u = x_u.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    sup = framework.compute_supervised_loss(x_l, y_l)
                    cons = framework.compute_consistency_loss(x_u)
                    loss = lambda_sup * sup + lambda_cons * cons
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sup = framework.compute_supervised_loss(x_l, y_l)
                cons = framework.compute_consistency_loss(x_u)
                loss = lambda_sup * sup + lambda_cons * cons
                loss.backward()
                optimizer.step()
            
            framework.update_teacher()
            progress.update(1)
        
        progress.close()