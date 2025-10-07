"""
Complete RLGSSL Training Algorithm
Implements Algorithm 1: Pseudo-Label Based Policy Gradient Descent
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from typing import Dict, Any, Optional, Tuple
import json
from tqdm import tqdm

# Import our custom modules
from data.ssl_datasets import create_ssl_dataloaders
from models.networks import create_model
from rlgssl.teacher_student import TeacherStudentFramework, warmup_teacher_student
from rlgssl.mixup import MixupGenerator
from rlgssl.reward_function import RLRewardFunction, AdaptiveRewardFunction
from rlgssl.rl_loss import RLLossFunction, AdaptiveRLLossFunction, CombinedLossFunction


class RLGSSLConfig:
    """Configuration class for RLGSSL training"""
    
    def __init__(self):
        # Dataset settings
        self.dataset_name = 'cifar10'
        self.num_labeled = 1000
        self.data_root = './data'
        
        # Model settings
        self.architecture = 'cnn13'  # 'cnn13' or 'wrn28'
        self.num_classes = 10
        self.dropout_rate = 0.0
        
        # Training settings
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.warmup_epochs = 50
        self.rlgssl_epochs = 400
        self.total_epochs = self.warmup_epochs + self.rlgssl_epochs
        
        # RLGSSL hyperparameters (from paper)
        self.lambda_sup = 0.1
        self.lambda_cons = 0.1
        self.ema_decay = 0.999
        self.mixup_alpha = 1.0
        
        # RL settings
        self.use_adaptive_rl = True
        self.confidence_threshold = 0.8
        self.entropy_regularization = 0.01
        
        # Evaluation and logging
        self.eval_frequency = 10
        self.save_frequency = 50
        self.log_frequency = 10
        
        # Device settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = 4
        self.use_amp = True
        
        # Experiment settings
        self.experiment_name = 'rlgssl_experiment'
        self.save_dir = './experiments'
        self.seed = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLGSSLConfig':
        """Load config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if key == 'device':
                setattr(config, key, torch.device(value))
            else:
                setattr(config, key, value)
        return config


class RLGSSLTrainer:
    """Complete RLGSSL Trainer implementing Algorithm 1"""
    
    def __init__(self, config: RLGSSLConfig):
        self.config = config
        self.device = config.device
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.seed)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_rlgssl_components()
        self._setup_optimizer()
        self._setup_logging()
        
        print(f"RLGSSL Trainer initialized on {self.device}")
        print(f"Dataset: {config.dataset_name}, Architecture: {config.architecture}")
        print(f"Labeled samples: {config.num_labeled}, Batch size: {config.batch_size}")
    
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Enable CuDNN benchmarking for fixed-size inputs to speed up convolutions
            torch.backends.cudnn.benchmark = True
    
    def _setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        dataloaders = create_ssl_dataloaders(
            dataset_name=self.config.dataset_name,
            num_labeled=self.config.num_labeled,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            seed=self.config.seed,
            data_root=self.config.data_root
        )
        
        self.train_labeled_loader = dataloaders['train_labeled']
        self.train_unlabeled_loader = dataloaders['train_unlabeled']
        self.test_loader = dataloaders['test']
        self.num_classes = dataloaders['num_classes']
        
        # Update config with actual number of classes
        self.config.num_classes = self.num_classes
    
    def _setup_model(self):
        """Setup student and teacher models"""
        print("Setting up models...")
        student_model = create_model(
            architecture=self.config.architecture,
            num_classes=self.num_classes,
            dropout_rate=self.config.dropout_rate
        )
        
        # Create teacher-student framework
        self.framework = TeacherStudentFramework(
            student_model=student_model,
            ema_decay=self.config.ema_decay,
            device=self.device
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in student_model.parameters())
        trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
        print(f"Model parameters: {trainable_params:,} trainable, {total_params:,} total")
    
    def _setup_rlgssl_components(self):
        """Setup RLGSSL-specific components"""
        print("Setting up RLGSSL components...")
        
        # Mixup generator
        self.mixup_generator = MixupGenerator(
            alpha=self.config.mixup_alpha,
            device=self.device
        )
        
        # Reward function
        if self.config.use_adaptive_rl:
            self.reward_function = AdaptiveRewardFunction(device=self.device)
            self.rl_loss_function = AdaptiveRLLossFunction(
                reward_function=self.reward_function,
                device=self.device,
                confidence_threshold=self.config.confidence_threshold,
                entropy_regularization=self.config.entropy_regularization
            )
        else:
            self.reward_function = RLRewardFunction(device=self.device)
            self.rl_loss_function = RLLossFunction(
                reward_function=self.reward_function,
                device=self.device
            )
        
        # Combined loss function
        self.combined_loss_function = CombinedLossFunction(
            rl_loss_fn=self.rl_loss_function,
            lambda_sup=self.config.lambda_sup,
            lambda_cons=self.config.lambda_cons,
            device=self.device
        )
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.Adam(
            self.framework.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.total_epochs,
            eta_min=1e-6
        )
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda' and self.config.use_amp))
    
    def _setup_logging(self):
        """Setup logging and experiment directory"""
        self.experiment_dir = os.path.join(self.config.save_dir, self.config.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.experiment_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Initialize logging
        self.training_logs = []
        self.best_accuracy = 0.0
        self.best_epoch = 0
    
    def warmup_phase(self):
        """Warmup training with standard teacher-student framework"""
        print(f"\n=== Starting Warmup Phase ({self.config.warmup_epochs} epochs) ===")
        
        warmup_teacher_student(
            framework=self.framework,
            labeled_loader=self.train_labeled_loader,
            unlabeled_loader=self.train_unlabeled_loader,
            optimizer=self.optimizer,
            num_epochs=self.config.warmup_epochs,
            lambda_sup=self.config.lambda_sup,
            lambda_cons=self.config.lambda_cons,
            scaler=self.scaler if hasattr(self, 'scaler') else None
        )
        
        # Evaluate after warmup
        warmup_results = self.framework.evaluate(self.test_loader)
        print(f"Warmup Results - Student Accuracy: {warmup_results['student_accuracy']:.2f}%")
        print(f"Warmup Results - Teacher Accuracy: {warmup_results['teacher_accuracy']:.2f}%")
        
        return warmup_results
    
    def rlgssl_training_step(
        self,
        labeled_batch,
        unlabeled_batch,
        epoch: int
    ) -> Dict[str, float]:
        """Single RLGSSL training step following Algorithm 1"""
        
        # Unpack batches robustly (support 2- or 3-tuples, or dicts)
        def unpack(batch):
            if isinstance(batch, dict):
                data = batch.get('data')
                labels = batch.get('labels')
                return data, labels
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    return batch[0], batch[1]
                elif len(batch) == 1:
                    return batch[0], None
            # Fallback
            return batch, None

        labeled_data, labels = unpack(labeled_batch)
        unlabeled_data, _ = unpack(unlabeled_batch)
        
        labeled_data = labeled_data.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        unlabeled_data = unlabeled_data.to(self.device, non_blocking=True)
        
        # Step 3: Compute soft pseudo-labels using teacher model
        autocast_enabled = (self.device.type == 'cuda' and self.config.use_amp)
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            pseudo_labels = self.framework.forward_teacher(unlabeled_data, return_logits=False)
        
        # Step 4: Generate mixup data
        labeled_batch_dict = {'data': labeled_data, 'labels': labels}
        unlabeled_batch_dict = {'data': unlabeled_data}
        
        mixup_result = self.mixup_generator.create_balanced_mixup_batch(
            labeled_batch_dict, unlabeled_batch_dict, pseudo_labels
        )
        
        mixup_data = mixup_result['data']
        mixup_labels = mixup_result['labels']
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Steps 7-8: Calculate reward and compute loss
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            if self.config.use_adaptive_rl and hasattr(self.rl_loss_function, 'compute_adaptive_rl_loss'):
                # Adaptive RL loss
                rl_loss_info = self.rl_loss_function.compute_adaptive_rl_loss(
                    model=self.framework.student_model,
                    unlabeled_data=unlabeled_data,
                    mixup_data=mixup_data,
                    mixup_labels=mixup_labels,
                    mu_values=mixup_result['mu_values'],
                    epoch=epoch - self.config.warmup_epochs,
                    max_epochs=self.config.rlgssl_epochs
                )
                
                # Compute supervised and consistency losses separately
                sup_loss = self.framework.compute_supervised_loss(labeled_data, labels)
                cons_loss = self.framework.compute_consistency_loss(unlabeled_data)
                
                total_loss = (
                    rl_loss_info['adaptive_rl_loss'] +
                    self.config.lambda_sup * sup_loss +
                    self.config.lambda_cons * cons_loss
                )
                
                loss_info = {
                    'total_loss': total_loss.detach(),
                    'rl_loss': rl_loss_info['adaptive_rl_loss'].detach(),
                    'supervised_loss': sup_loss.detach(),
                    'consistency_loss': cons_loss.detach(),
                    'reward': rl_loss_info.get('base_reward', torch.tensor(0.0)).detach(),
                    'confidence_ratio': rl_loss_info.get('confidence_ratio', torch.tensor(0.0)).detach()
                }
            else:
                # Standard combined loss
                loss_info = self.combined_loss_function.compute_combined_loss(
                    framework=self.framework,
                    labeled_data=labeled_data,
                    labels=labels,
                    unlabeled_data=unlabeled_data,
                    mixup_data=mixup_data,
                    mixup_labels=mixup_labels
                )
                total_loss = loss_info['total_loss']
        
        # Step 9: Update student parameters (with AMP if enabled)
        if self.scaler.is_enabled():
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()
        
        # Step 10: Update teacher parameters via EMA
        self.framework.update_teacher()
        
        # Move logged tensors to CPU scalars for printing
        for key, value in list(loss_info.items()):
            if isinstance(value, torch.Tensor):
                try:
                    loss_info[key] = value.item()
                except Exception:
                    loss_info[key] = float(value.detach().cpu())
        
        return loss_info
    
    def rlgssl_phase(self):
        """RLGSSL training phase"""
        print(f"\n=== Starting RLGSSL Phase ({self.config.rlgssl_epochs} epochs) ===")
        
        self.framework.student_model.train()
        
        for epoch in range(self.config.warmup_epochs, self.config.total_epochs):
            epoch_start_time = time.time()
            epoch_losses = {
                'total_loss': 0.0,
                'rl_loss': 0.0,
                'supervised_loss': 0.0,
                'consistency_loss': 0.0,
                'reward': 0.0
            }
            
            # Create data iterators
            labeled_iter = iter(self.train_labeled_loader)
            unlabeled_iter = iter(self.train_unlabeled_loader)
            
            num_batches = 0
            progress_bar = tqdm(
                desc=f"Epoch {epoch+1}/{self.config.total_epochs}",
                total=min(len(self.train_labeled_loader), len(self.train_unlabeled_loader))
            )
            
            # Training loop for one epoch
            while True:
                try:
                    labeled_batch = next(labeled_iter)
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    break
                
                # RLGSSL training step
                step_losses = self.rlgssl_training_step(labeled_batch, unlabeled_batch, epoch)
                
                # Accumulate losses
                for key, value in step_losses.items():
                    if key in epoch_losses:
                        epoch_losses[key] += value
                
                num_batches += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'Loss': f"{step_losses['total_loss']:.4f}",
                    'Reward': f"{step_losses.get('reward', 0):.4f}"
                })
            
            progress_bar.close()
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
            
            # Update learning rate
            self.scheduler.step()
            
            # Evaluation
            if (epoch + 1) % self.config.eval_frequency == 0:
                eval_results = self.framework.evaluate(self.test_loader)
                
                # Log results
                log_entry = {
                    'epoch': epoch + 1,
                    'phase': 'rlgssl',
                    'training_time': time.time() - epoch_start_time,
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    **epoch_losses,
                    **eval_results
                }
                
                self.training_logs.append(log_entry)
                
                # Print progress
                print(f"Epoch {epoch+1}: "
                      f"Loss={epoch_losses['total_loss']:.4f}, "
                      f"RL={epoch_losses['rl_loss']:.4f}, "
                      f"Reward={epoch_losses['reward']:.4f}, "
                      f"Acc={eval_results['student_accuracy']:.2f}%")
                
                # Save best model
                if eval_results['student_accuracy'] > self.best_accuracy:
                    self.best_accuracy = eval_results['student_accuracy']
                    self.best_epoch = epoch + 1
                    self.save_checkpoint('best_model.pth', epoch + 1, eval_results)
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_frequency == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch + 1)
    
    def train(self):
        """Complete RLGSSL training procedure"""
        print("Starting RLGSSL Training...")
        start_time = time.time()
        
        try:
            # Phase 1: Warmup with standard SSL
            warmup_results = self.warmup_phase()
            
            # Phase 2: RLGSSL training
            self.rlgssl_phase()
            
            # Final evaluation
            final_results = self.framework.evaluate(self.test_loader)
            
            total_time = time.time() - start_time
            print(f"\n=== Training Completed ===")
            print(f"Total training time: {total_time/3600:.2f} hours")
            print(f"Best accuracy: {self.best_accuracy:.2f}% (epoch {self.best_epoch})")
            print(f"Final accuracy: {final_results['student_accuracy']:.2f}%")
            
            # Save final results
            final_summary = {
                'config': self.config.to_dict(),
                'warmup_results': warmup_results,
                'final_results': final_results,
                'best_accuracy': self.best_accuracy,
                'best_epoch': self.best_epoch,
                'total_training_time': total_time,
                'training_logs': self.training_logs
            }
            
            summary_path = os.path.join(self.experiment_dir, 'training_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(final_summary, f, indent=2, default=str)
            
            return final_summary
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self.save_checkpoint('interrupted_checkpoint.pth', -1)
            return None
        except Exception as e:
            print(f"\nTraining failed with error: {e}")
            raise e
    
    def save_checkpoint(self, filename: str, epoch: int, eval_results: Optional[Dict] = None):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'student_state_dict': self.framework.student_model.state_dict(),
            'teacher_state_dict': self.framework.teacher.teacher_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'best_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'training_logs': self.training_logs
        }
        
        if eval_results:
            checkpoint['eval_results'] = eval_results
        
        checkpoint_path = os.path.join(self.experiment_dir, filename)
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.framework.student_model.load_state_dict(checkpoint['student_state_dict'])
        self.framework.teacher.teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        self.best_epoch = checkpoint.get('best_epoch', 0)
        self.training_logs = checkpoint.get('training_logs', [])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def main():
    """Main training function"""
    # Create configuration
    config = RLGSSLConfig()
    
    # Customize config for different experiments
    # config.dataset_name = 'cifar100'
    # config.num_labeled = 4000
    # config.architecture = 'wrn28'
    
    # Create trainer and start training
    trainer = RLGSSLTrainer(config)
    results = trainer.train()
    
    if results:
        print("\nTraining completed successfully!")
        print(f"Results saved to: {trainer.experiment_dir}")
    else:
        print("\nTraining was interrupted or failed.")


if __name__ == "__main__":
    main()
