"""
Neural Network Architectures for RLGSSL
Implements CNN-13 and WRN-28 as described in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List


class CNN13(nn.Module):
    """
    13-layer CNN architecture for CIFAR-10/100 and SVHN
    Based on the architecture described in the RLGSSL paper
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(CNN13, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Block 2
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        # Block 3
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=0)  # No padding to reduce size
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.conv9 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn9 = nn.BatchNorm2d(128)
        
        # Additional layers to reach 13 layers
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(128)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.25, training=self.training)
        
        # Block 2
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.1)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.25, training=self.training)
        
        # Block 3
        x = F.leaky_relu(self.bn7(self.conv7(x)), 0.1)
        x = F.leaky_relu(self.bn8(self.conv8(x)), 0.1)
        x = F.leaky_relu(self.bn9(self.conv9(x)), 0.1)
        
        # Additional layers
        x = F.leaky_relu(self.bn10(self.conv10(x)), 0.1)
        x = F.leaky_relu(self.bn11(self.conv11(x)), 0.1)
        x = F.leaky_relu(self.bn12(self.conv12(x)), 0.1)
        
        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic block for Wide ResNet"""
    
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, dropout_rate: float = 0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                              padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, 
                              padding=1, bias=False)
        
        self.dropout_rate = dropout_rate
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        
        out = self.relu2(self.bn2(out))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(out)
        
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """
    Wide ResNet architecture
    Based on "Wide Residual Networks" paper by Zagoruyko & Komodakis
    """
    
    def __init__(self, depth: int, widen_factor: int, num_classes: int = 10, 
                 dropout_rate: float = 0.0):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor
        
        print(f'| Wide-Resnet {depth}x{k}')
        
        n_stages = [16, 16*k, 32*k, 64*k]
        
        self.conv1 = nn.Conv2d(3, n_stages[0], kernel_size=3, stride=1, 
                              padding=1, bias=False)
        
        self.layer1 = self._wide_layer(BasicBlock, n_stages[1], n, stride=1, 
                                      dropout_rate=dropout_rate)
        self.layer2 = self._wide_layer(BasicBlock, n_stages[2], n, stride=2, 
                                      dropout_rate=dropout_rate)
        self.layer3 = self._wide_layer(BasicBlock, n_stages[3], n, stride=2, 
                                      dropout_rate=dropout_rate)
        
        self.bn1 = nn.BatchNorm2d(n_stages[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_stages[3], num_classes)
        
        self._initialize_weights()
    
    def _wide_layer(self, block, planes: int, num_blocks: int, stride: int, 
                   dropout_rate: float):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


def WRN28(num_classes: int = 10, dropout_rate: float = 0.0) -> WideResNet:
    """
    Wide ResNet-28 with width factor 2 (WRN-28-2)
    This is the architecture used in the RLGSSL paper
    """
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes, 
                     dropout_rate=dropout_rate)


class PolicyNetwork(nn.Module):
    """
    Wrapper for policy networks used in RLGSSL
    Can use either CNN-13 or WRN-28 as backbone
    Returns probabilities for RL formulation
    """
    
    def __init__(self, backbone: str, num_classes: int = 10, dropout_rate: float = 0.0):
        super(PolicyNetwork, self).__init__()
        self.backbone_name = backbone.lower()
        self.num_classes = num_classes
        
        if self.backbone_name == 'cnn13':
            self.backbone = CNN13(num_classes=num_classes, dropout_rate=dropout_rate)
        elif self.backbone_name == 'wrn28':
            self.backbone = WRN28(num_classes=num_classes, dropout_rate=dropout_rate)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass that returns probabilities (for RL) or logits (for standard training)
        
        Args:
            x: Input tensor
            return_logits: If True, return raw logits; if False, return probabilities
        
        Returns:
            Probabilities (softmax) or logits
        """
        logits = self.backbone(x)
        
        if return_logits:
            return logits
        else:
            # Return probabilities for RL formulation
            return F.softmax(logits, dim=1)
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits (for computing losses)"""
        return self.forward(x, return_logits=True)
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Get probabilities (for RL actions)"""
        return self.forward(x, return_logits=False)


def create_model(architecture: str, num_classes: int, dropout_rate: float = 0.0) -> PolicyNetwork:
    """
    Factory function to create models
    
    Args:
        architecture: 'cnn13' or 'wrn28'
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
    
    Returns:
        PolicyNetwork instance
    """
    return PolicyNetwork(architecture, num_classes, dropout_rate)


# Model testing and information
def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Testing CNN-13...")
    cnn13 = create_model('cnn13', num_classes=10)
    print(f"CNN-13 parameters: {count_parameters(cnn13):,}")
    
    print("\nTesting WRN-28...")
    wrn28 = create_model('wrn28', num_classes=10)
    print(f"WRN-28 parameters: {count_parameters(wrn28):,}")
    
    # Test forward pass
    test_input = torch.randn(2, 3, 32, 32)
    
    print(f"\nTest input shape: {test_input.shape}")
    
    with torch.no_grad():
        cnn13_logits = cnn13.get_logits(test_input)
        cnn13_probs = cnn13.get_probabilities(test_input)
        print(f"CNN-13 logits shape: {cnn13_logits.shape}")
        print(f"CNN-13 probabilities shape: {cnn13_probs.shape}")
        print(f"CNN-13 probabilities sum: {cnn13_probs.sum(dim=1)}")
        
        wrn28_logits = wrn28.get_logits(test_input)
        wrn28_probs = wrn28.get_probabilities(test_input)
        print(f"WRN-28 logits shape: {wrn28_logits.shape}")
        print(f"WRN-28 probabilities shape: {wrn28_probs.shape}")
        print(f"WRN-28 probabilities sum: {wrn28_probs.sum(dim=1)}")
