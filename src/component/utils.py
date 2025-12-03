import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def load_source_data(num_labeled, num_unlabeled, num_validation):
    """
    Load and split MNIST dataset into labeled, unlabeled, and validation sets.

    Args:
        num_labeled (int): Number of labeled samples
        num_unlabeled (int): Number of unlabeled samples
        num_validation (int): Number of validation samples

    Returns:
        tuple: (labeled_data, unlabeled_data, validation_data)
               Each item is a tuple of (features, labels)
    """
    transform = transforms.ToTensor()
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    all_data = mnist_train.data.float() / 255.0
    all_targets = mnist_train.targets

    # Split validation set first
    data_temp, val_data, targets_temp, val_targets = train_test_split(
        all_data, all_targets, test_size=num_validation, stratify=all_targets, random_state=42
    )

    # Then split labeled vs. unlabeled
    labeled_x, unlabeled_x, labeled_y, unlabeled_y_true = train_test_split(
        data_temp, targets_temp, train_size=num_labeled, test_size=num_unlabeled,
        stratify=targets_temp, random_state=42
    )

    unique_classes = torch.unique(labeled_y)
    expected_classes = set(range(10))  # MNIST classes 0–9
    missing = expected_classes - set(unique_classes.tolist())
    if missing:
        print(f" Missing classes in labeled data: {missing}")

    labeled_data = (labeled_x.view(-1, 784), labeled_y)
    unlabeled_data = (unlabeled_x.view(-1, 784), unlabeled_y_true)
    validation_data = (val_data.view(-1, 784), val_targets)

    return labeled_data, unlabeled_data, validation_data
