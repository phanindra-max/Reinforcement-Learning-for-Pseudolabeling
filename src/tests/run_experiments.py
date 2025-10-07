"""
Experiment Runner for RLGSSL
Reproduces the experimental setup from the paper
"""

import argparse
import os
import sys
from datetime import datetime
import torch

from train_rlgssl import RLGSSLConfig, RLGSSLTrainer


def create_paper_configs():
    """
    Create configurations that match the paper's experimental setup
    
    Paper experiments:
    - CIFAR-10: 1000, 2000, 4000 labels with CNN-13 and WRN-28
    - CIFAR-100: 4000, 10000 labels with CNN-13 and WRN-28  
    - SVHN: 500, 1000 labels with CNN-13 and WRN-28
    """
    
    configs = []
    
    # CIFAR-10 experiments
    for architecture in ['cnn13', 'wrn28']:
        for num_labeled in [1000, 2000, 4000]:
            config = RLGSSLConfig()
            config.dataset_name = 'cifar10'
            config.num_labeled = num_labeled
            config.architecture = architecture
            config.num_classes = 10
            config.experiment_name = f'cifar10_{architecture}_{num_labeled}labels_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            configs.append(config)
    
    # CIFAR-100 experiments
    for architecture in ['cnn13', 'wrn28']:
        for num_labeled in [4000, 10000]:
            config = RLGSSLConfig()
            config.dataset_name = 'cifar100'
            config.num_labeled = num_labeled
            config.architecture = architecture
            config.num_classes = 100
            config.experiment_name = f'cifar100_{architecture}_{num_labeled}labels_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            configs.append(config)
    
    # SVHN experiments
    for architecture in ['cnn13', 'wrn28']:
        for num_labeled in [500, 1000]:
            config = RLGSSLConfig()
            config.dataset_name = 'svhn'
            config.num_labeled = num_labeled
            config.architecture = architecture
            config.num_classes = 10
            config.experiment_name = f'svhn_{architecture}_{num_labeled}labels_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            configs.append(config)
    
    return configs


def run_single_experiment(config: RLGSSLConfig, experiment_id: str):
    """Run a single experiment with the given configuration"""
    
    print(f"\n{'='*80}")
    print(f"Starting Experiment {experiment_id}")
    print(f"Dataset: {config.dataset_name}, Architecture: {config.architecture}")
    print(f"Labeled samples: {config.num_labeled}")
    print(f"{'='*80}")
    
    try:
        # Create trainer
        trainer = RLGSSLTrainer(config)
        
        # Run training
        results = trainer.train()
        
        if results:
            final_accuracy = results['final_results']['student_accuracy']
            best_accuracy = results['best_accuracy']
            
            print(f"\n{'='*80}")
            print(f"Experiment {experiment_id} COMPLETED")
            print(f"Final Accuracy: {final_accuracy:.2f}%")
            print(f"Best Accuracy: {best_accuracy:.2f}% (Epoch {results['best_epoch']})")
            print(f"Results saved to: {trainer.experiment_dir}")
            print(f"{'='*80}")
            
            return {
                'experiment_id': experiment_id,
                'config': config.to_dict(),
                'final_accuracy': final_accuracy,
                'best_accuracy': best_accuracy,
                'best_epoch': results['best_epoch'],
                'experiment_dir': trainer.experiment_dir,
                'status': 'completed'
            }
        else:
            print(f"Experiment {experiment_id} FAILED or INTERRUPTED")
            return {
                'experiment_id': experiment_id,
                'config': config.to_dict(),
                'status': 'failed'
            }
            
    except Exception as e:
        print(f"Experiment {experiment_id} FAILED with error: {e}")
        return {
            'experiment_id': experiment_id,
            'config': config.to_dict(),
            'error': str(e),
            'status': 'error'
        }


def run_paper_reproduction():
    """Run all experiments to reproduce paper results"""
    
    print("RLGSSL Paper Reproduction Experiments")
    print("=====================================")
    
    # Get all configurations
    configs = create_paper_configs()
    
    print(f"Total experiments to run: {len(configs)}")
    
    # Run experiments
    all_results = []
    for i, config in enumerate(configs, 1):
        experiment_id = f"{i:02d}_{config.dataset_name}_{config.architecture}_{config.num_labeled}"
        
        result = run_single_experiment(config, experiment_id)
        all_results.append(result)
        
        # Save intermediate results
        import json
        results_file = f"paper_reproduction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*80)
    print("PAPER REPRODUCTION SUMMARY")
    print("="*80)
    
    successful_experiments = [r for r in all_results if r['status'] == 'completed']
    failed_experiments = [r for r in all_results if r['status'] != 'completed']
    
    print(f"Successful experiments: {len(successful_experiments)}/{len(all_results)}")
    
    if successful_experiments:
        print("\nSuccessful Experiments:")
        print("-" * 50)
        for result in successful_experiments:
            config = result['config']
            print(f"{config['dataset_name']} {config['architecture']} {config['num_labeled']}L: "
                  f"Best={result['best_accuracy']:.2f}%, Final={result['final_accuracy']:.2f}%")
    
    if failed_experiments:
        print(f"\nFailed experiments: {len(failed_experiments)}")
        for result in failed_experiments:
            config = result['config']
            print(f"FAILED: {config['dataset_name']} {config['architecture']} {config['num_labeled']}L")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return all_results


def run_quick_test():
    """Run a quick test with minimal settings"""
    print("Running Quick Test...")
    
    config = RLGSSLConfig()
    config.dataset_name = 'cifar10'
    config.num_labeled = 1000
    config.architecture = 'cnn13'
    config.warmup_epochs = 5  # Reduced for quick test
    config.rlgssl_epochs = 20  # Reduced for quick test
    config.eval_frequency = 5
    config.experiment_name = f'quick_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    return run_single_experiment(config, "quick_test")


def main():
    parser = argparse.ArgumentParser(description='RLGSSL Experiment Runner')
    parser.add_argument('--mode', choices=['paper', 'quick', 'single'], default='single',
                        help='Experiment mode: paper (full reproduction), quick (fast test), single (custom)')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar100', 'svhn'], default='cifar10',
                        help='Dataset for single experiment')
    parser.add_argument('--architecture', choices=['cnn13', 'wrn28'], default='cnn13',
                        help='Architecture for single experiment')
    parser.add_argument('--num_labeled', type=int, default=1000,
                        help='Number of labeled samples for single experiment')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available. Using CPU.")
    
    if args.mode == 'paper':
        print("Running full paper reproduction experiments...")
        print("WARNING: This will take a very long time (days)!")
        confirm = input("Continue? (y/N): ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return
        
        results = run_paper_reproduction()
        
    elif args.mode == 'quick':
        print("Running quick test...")
        result = run_quick_test()
        print(f"Quick test result: {result}")
        
    else:  # single experiment
        print("Running single experiment...")
        config = RLGSSLConfig()
        config.dataset_name = args.dataset
        config.architecture = args.architecture
        config.num_labeled = args.num_labeled
        config.experiment_name = f'{args.dataset}_{args.architecture}_{args.num_labeled}labels_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
        # Set number of classes based on dataset
        if args.dataset == 'cifar100':
            config.num_classes = 100
        else:
            config.num_classes = 10
        
        result = run_single_experiment(config, "single_experiment")
        print(f"Single experiment result: {result}")


if __name__ == "__main__":
    main()
