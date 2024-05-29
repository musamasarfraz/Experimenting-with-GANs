import torch
from torch import nn

# Base class for generators
class ConditionalGenerator(nn.Module):
    def __init__(self, n_classes):
        super(ConditionalGenerator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, 10)
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        x = self.model(x)
        return x.view(x.size(0), 1, 28, 28)

# Specific generator for uppercase letters
class ConditionalGeneratorUppercase(ConditionalGenerator):
    def __init__(self, n_classes=26):
        super(ConditionalGeneratorUppercase, self).__init__(n_classes)

# Specific generator for lowercase letters
class ConditionalGeneratorLowercase(ConditionalGenerator):
    def __init__(self, n_classes=26):
        super(ConditionalGeneratorLowercase, self).__init__(n_classes)

# Specific generator for digits
class ConditionalGeneratorDigits(ConditionalGenerator):
    def __init__(self, n_classes=10):
        super(ConditionalGeneratorDigits, self).__init__(n_classes)
