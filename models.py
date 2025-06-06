import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


#GIA, GRNA-MNIST, GRNA-DRIVE
class APModel(nn.Module):
    def __init__(self, in_features, hidden_layer, out_features):
        super(APModel, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PPModel(nn.Module):
    def __init__(self, in_features, hidden_layer, out_features):
        super(PPModel, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LearningCoordinator(nn.Module):
    def __init__(self, apfeatures, ppfeatures, num_classes):
        super(LearningCoordinator, self).__init__()
        self.fc1 = nn.Linear(apfeatures+ppfeatures, num_classes)

    def forward(self, apfeatures, ppfeatures):
        x = torch.cat((apfeatures, ppfeatures), dim=1)
        x = self.fc1(x)
        confidence_scores = F.softmax(x, dim=1)
        return confidence_scores
    
#GIA, GRNA - CIFAR10, CIFAR100
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class LearningCoordinatorCIFAR(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(x)
        return logits
    
class BottomModel(nn.Module):
    def __init__(self, out_dim=64, gpu=True):
        super(BottomModel, self).__init__()
        self.gpu = gpu
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # ResNet-20 consists of 3 stages, each with 3 residual blocks
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size (batch, 64, 1, 1)
        self.fc = nn.Linear(64, out_dim)  # Final output layer with out_dim=64

        if gpu:
            self.cuda()

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # First block handles stride, others are stride=1
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.gpu:
            x = x.cuda()

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = torch.flatten(out, 1)  
        out = self.fc(out)  
        return out
    
#GRNA ADULT
class APModelADULT(nn.Module):
    def __init__(self, in_features, hidden_layer, out_features):
        super(APModelADULT, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PPModelADULT(nn.Module):
    def __init__(self, in_features, hidden_layer, out_features):
        super(PPModelADULT, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LearningCoordinatorADULT(nn.Module):
    def __init__(self, apfeatures, ppfeatures, num_classes):
        super(LearningCoordinatorADULT, self).__init__()
        self.fc1 = nn.Linear(apfeatures+ppfeatures, num_classes)

    def forward(self, apfeatures, ppfeatures):
        x = torch.cat((apfeatures, ppfeatures), dim=1)
        x = self.fc1(x)
        return x
    
class VFLDataset(Dataset):
    def __init__(self, features, labels, percent_active):
        self.x = features.astype(np.float32)
        self.y = labels.astype(np.int64)

        D = self.x.shape[1]
        self.num_active = max(1, int(D * percent_active / 100.0))

        self.x_act = self.x[:, :self.num_active]
        self.x_pas = self.x[:, self.num_active:]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_act[idx], self.x_pas[idx], self.y[idx]

class VFLLogistic(nn.Module):
    def __init__(self, in_act, in_pas, num_classes):
        super().__init__()
        self.linear_act = nn.Linear(in_act,  num_classes, bias=False)
        self.linear_pas = nn.Linear(in_pas,  num_classes, bias=False)
        self.softmax    = nn.Softmax(dim=1)

    def forward(self, x_act, x_pas):
        z_act  = self.linear_act(x_act)
        z_pas  = self.linear_pas(x_pas)
        logits = z_act + z_pas
        return self.softmax(logits), logits

