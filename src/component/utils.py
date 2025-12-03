import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from config import *

def load_source_data(data_source, num_labeled, num_unlabeled, num_validation):
    print(f"{'=' * 70}")
    print("DEVICE CONFIGURATION")
    print(f"{'=' * 70}")
    print(f"Using device: {device}\n")
    print(f"{'=' * 70}")
    print("DATASET LOADING AND PREPERATION")
    print(f"{'=' * 70}")
    print(f"Loading {data_source.upper()} dataset...")
    transform = transforms.ToTensor()

    if data_source.lower() == "mnist":
        # Load Train Dataset
        dataset_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_data = dataset_train.data.float() / 255.0
        train_targets = dataset_train.targets
        # Load Test Dataset
        dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_data = dataset_test.data.float() / 255.0
        test_y = dataset_test.targets
        # (N, H, W) -> (N, C, H, W) with C=1
        train_data = train_data.unsqueeze(1)
        test_data = test_data.unsqueeze(1)

    elif data_source.lower() == "cifar10":
        # Load Train Dataset
        dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_data = torch.from_numpy(dataset_train.data).float() / 255.0
        train_targets = torch.tensor(dataset_train.targets)
        # Load Test Dataset
        dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_data =  torch.from_numpy(dataset_test.data).float() / 255.0
        test_y = torch.tensor(dataset_test.targets)
        # (N, H, W, C) → (N, C, H, W)
        train_data = train_data.permute(0, 3, 1, 2)
        test_data = test_data.permute(0, 3, 1, 2)

    elif data_source.lower() == "cifar100":
        # Load Train Dataset
        dataset_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        train_data = torch.from_numpy(dataset_train.data).float() / 255.0
        train_targets = torch.tensor(dataset_train.targets)
        # Load Test Dataset
        dataset_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        test_data = torch.from_numpy(dataset_test.data).float() / 255.0
        test_y = torch.tensor(dataset_test.targets)
        # (N, H, W, C) → (N, C, H, W)
        train_data = train_data.permute(0, 3, 1, 2)
        test_data = test_data.permute(0, 3, 1, 2)

    else:
        raise ValueError(f"Invalid dataset name: {data_source}")

    # Metadata extraction
    num_classes = len(torch.unique(train_targets))
    input_shape = tuple(train_data.shape[1:])
    feature_dim = int(torch.prod(torch.tensor(input_shape)).item())

    dataset_info = {
        'num_classes': num_classes,
        'feature_dim': feature_dim,
        'input_shape': input_shape
    }
    # Data size validatiosn before split
    if(len(train_data) < num_labeled + num_unlabeled + num_validation):
        raise ValueError(f"Not enough data for the requested split. Please reduce the number of labeled/unlabeled/validation samples so that they add up to {len(train_data)}.")

    # Data Split
    data_temp, unlabeled_data, targets_temp, unlabeled_y = train_test_split(train_data, train_targets, test_size = num_unlabeled, stratify=train_targets, random_state=42)

    labeled_data, val_data, labeled_y, val_y = train_test_split(data_temp, targets_temp, train_size=num_labeled, test_size=num_validation, stratify=targets_temp, random_state=42)

    # Flatten Data
    labeled_x = labeled_data.reshape(labeled_data.size(0), -1)
    unlabeled_x = unlabeled_data.reshape(unlabeled_data.size(0), -1)
    val_x = val_data.reshape(val_data.size(0), -1)
    test_x = test_data.reshape(test_data.size(0), -1)

    print("\nDataset Summary:")
    print("-" * 45)
    print(f"{'Dataset':<25} {'Size':>10}")
    print("-" * 45)
    print(f"{'Train set':<25} {len(train_data):>10,}")
    print(f"{'Test set':<25} {len(test_data):>10,}")
    print(f"{'Labeled data':<25} {len(labeled_data):>10,}")
    print(f"{'Unlabeled data':<25} {len(unlabeled_data):>10,}")
    print(f"{'Validation data':<25} {len(val_data):>10,}")
    print(f"{'Discarded/Unused data':<25} {(len(train_data) - len(labeled_data) - len(unlabeled_data) - len(val_data)):>10,}")
    print("-" * 45)

    print("\nMetadata Information:")
    print("-" * 45)
    print(f"{'Property':<25} {'Value'}")
    print("-" * 45)
    print(f"{'Number of classes':<25} {num_classes}")
    print(f"{'Input shape':<25} {input_shape}")
    print(f"{'Feature dimension':<25} {feature_dim}")
    print(f"{'Flattened shape':<25} {labeled_x[0].shape}")
    print("-" * 45)

    return labeled_x, labeled_y, unlabeled_x, unlabeled_y, val_x, val_y, test_x, test_y, dataset_info