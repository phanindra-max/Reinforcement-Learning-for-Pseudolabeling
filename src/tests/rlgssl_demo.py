"""
RLGSSL Demo Script
Quick demonstration of the implemented RLGSSL method
"""

import torch
import numpy as np
from train_rlgssl import RLGSSLConfig, RLGSSLTrainer


def run_minimal_demo():
    """
    Run a minimal demo to verify the implementation works
    Uses very small settings for quick execution
    """
    
    print("RLGSSL Implementation Demo")
    print("=" * 50)
    
    # Create minimal config for quick demo
    config = RLGSSLConfig()
    config.dataset_name = 'cifar10'
    config.num_labeled = 1000
    config.architecture = 'cnn13'
    config.batch_size = 64  # Smaller batch size for demo
    config.warmup_epochs = 2  # Very short for demo
    config.rlgssl_epochs = 3   # Very short for demo
    config.eval_frequency = 1  # Evaluate every epoch
    config.experiment_name = 'demo_run'
    
    print(f"Configuration:")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Architecture: {config.architecture}")
    print(f"  Labeled samples: {config.num_labeled}")
    print(f"  Warmup epochs: {config.warmup_epochs}")
    print(f"  RLGSSL epochs: {config.rlgssl_epochs}")
    print(f"  Device: {config.device}")
    
    try:
        # Create trainer
        print("\nInitializing RLGSSL trainer...")
        trainer = RLGSSLTrainer(config)
        
        print("✓ Data loaders created successfully")
        print("✓ Models initialized successfully")
        print("✓ RLGSSL components set up successfully")
        
        # Test individual components
        print("\nTesting individual components...")
        
        # Test a single training step
        print("Testing single training step...")
        labeled_iter = iter(trainer.train_labeled_loader)
        unlabeled_iter = iter(trainer.train_unlabeled_loader)
        
        labeled_batch = next(labeled_iter)
        unlabeled_batch = next(unlabeled_iter)
        
        # Test one RLGSSL step
        trainer.framework.student_model.train()
        loss_info = trainer.rlgssl_training_step(labeled_batch, unlabeled_batch, epoch=0)
        
        print("✓ Single training step completed")
        print(f"  Total loss: {loss_info['total_loss']:.4f}")
        print(f"  RL loss: {loss_info['rl_loss']:.4f}")
        print(f"  Reward: {loss_info['reward']:.4f}")
        
        # Test evaluation
        print("Testing evaluation...")
        eval_results = trainer.framework.evaluate(trainer.test_loader)
        print("✓ Evaluation completed")
        print(f"  Initial accuracy: {eval_results['student_accuracy']:.2f}%")
        
        # Run short training
        print(f"\nRunning short training ({config.warmup_epochs + config.rlgssl_epochs} epochs total)...")
        
        # Run warmup
        print("Warmup phase...")
        warmup_results = trainer.warmup_phase()
        print(f"✓ Warmup completed - Accuracy: {warmup_results['student_accuracy']:.2f}%")
        
        # Run RLGSSL phase  
        print("RLGSSL phase...")
        trainer.rlgssl_phase()
        
        # Final evaluation
        final_results = trainer.framework.evaluate(trainer.test_loader)
        print(f"✓ RLGSSL completed - Final accuracy: {final_results['student_accuracy']:.2f}%")
        
        print("\n" + "=" * 50)
        print("DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 50)
        print(f"Accuracy improvement: {warmup_results['student_accuracy']:.2f}% → {final_results['student_accuracy']:.2f}%")
        print(f"Best accuracy achieved: {trainer.best_accuracy:.2f}%")
        print(f"Results saved to: {trainer.experiment_dir}")
        
        return {
            'initial_accuracy': eval_results['student_accuracy'],
            'warmup_accuracy': warmup_results['student_accuracy'],
            'final_accuracy': final_results['student_accuracy'],
            'best_accuracy': trainer.best_accuracy,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}


def test_components_individually():
    """Test individual components to verify they work correctly"""
    
    print("\nTesting Individual Components")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        from data.ssl_datasets import create_ssl_dataloaders
        
        dataloaders = create_ssl_dataloaders(
            dataset_name='cifar10',
            num_labeled=100,  # Very small for quick test
            batch_size=16
        )
        print("✓ Data loading works")
        
        # Test model creation
        print("2. Testing model creation...")
        from models.networks import create_model
        
        model = create_model('cnn13', num_classes=10)
        test_input = torch.randn(2, 3, 32, 32).to(device)
        output = model.get_probabilities(test_input)
        assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
        print("✓ Model creation and forward pass work")
        
        # Test teacher-student framework
        print("3. Testing teacher-student framework...")
        from rlgssl.teacher_student import TeacherStudentFramework
        
        framework = TeacherStudentFramework(model, device=device)
        test_data = torch.randn(4, 3, 32, 32).to(device)
        test_labels = torch.randint(0, 10, (4,)).to(device)
        
        sup_loss = framework.compute_supervised_loss(test_data, test_labels)
        cons_loss = framework.compute_consistency_loss(test_data)
        assert sup_loss.item() > 0, "Supervised loss should be positive"
        assert cons_loss.item() >= 0, "Consistency loss should be non-negative"
        print("✓ Teacher-student framework works")
        
        # Test mixup
        print("4. Testing mixup functionality...")
        from rlgssl.mixup import MixupGenerator
        
        mixup_gen = MixupGenerator(device=device)
        labeled_data = torch.randn(4, 3, 32, 32).to(device)
        labeled_labels = torch.randint(0, 10, (4,)).to(device)
        unlabeled_data = torch.randn(8, 3, 32, 32).to(device)
        pseudo_labels = torch.softmax(torch.randn(8, 10), dim=1).to(device)
        
        labeled_batch = {'data': labeled_data, 'labels': labeled_labels}
        unlabeled_batch = {'data': unlabeled_data}
        
        mixup_result = mixup_gen.create_balanced_mixup_batch(
            labeled_batch, unlabeled_batch, pseudo_labels
        )
        
        assert mixup_result['data'].shape[0] == 8, "Mixup data should match unlabeled size"
        assert mixup_result['labels'].shape == (8, 10), "Mixup labels should be (8, 10)"
        print("✓ Mixup functionality works")
        
        # Test reward function
        print("5. Testing reward function...")
        from rlgssl.reward_function import RLRewardFunction
        
        reward_fn = RLRewardFunction(device=device)
        reward = reward_fn.compute_reward(
            model, mixup_result['data'], mixup_result['labels']
        )
        assert isinstance(reward, torch.Tensor), "Reward should be a tensor"
        print("✓ Reward function works")
        
        # Test RL loss
        print("6. Testing RL loss function...")
        from rlgssl.rl_loss import RLLossFunction
        
        rl_loss_fn = RLLossFunction(reward_fn, device=device)
        loss_info = rl_loss_fn.compute_rl_loss(
            model, unlabeled_data, mixup_result['data'], mixup_result['labels']
        )
        
        assert 'rl_loss' in loss_info, "RL loss should be computed"
        assert isinstance(loss_info['rl_loss'], torch.Tensor), "RL loss should be a tensor"
        print("✓ RL loss function works")
        
        print("\n✅ All individual components working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting RLGSSL Implementation Demo\n")
    
    # Test individual components first
    components_ok = test_components_individually()
    
    if components_ok:
        print("\n" + "=" * 60)
        print("All components verified! Running full demo...")
        print("=" * 60)
        
        # Run minimal demo
        demo_results = run_minimal_demo()
        
        if demo_results['status'] == 'success':
            print("\n🎉 RLGSSL implementation is working correctly!")
            print("You can now run full experiments with:")
            print("  python run_experiments.py --mode quick")
            print("  python run_experiments.py --mode single --dataset cifar10 --num_labeled 1000")
        else:
            print("\n❌ Demo failed. Please check the error messages above.")
    else:
        print("\n❌ Component tests failed. Please fix the issues before running the full demo.")
