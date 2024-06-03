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


class ConditionalVAE(nn.Module):
    def __init__(self, z_dim=32, num_classes=62, img_size=28, image_channels=1):
        super(ConditionalVAE, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.fc1 = nn.Linear(image_channels * img_size * img_size + num_classes, 400)
        self.fc21 = nn.Linear(400, z_dim)  # Mean μ
        self.fc22 = nn.Linear(400, z_dim)  # Log variance σ
        self.fc3 = nn.Linear(z_dim + num_classes, 400)
        self.fc4 = nn.Linear(400, image_channels * img_size * img_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.class_embedding = nn.Embedding(num_classes, num_classes)

    def encode(self, x, labels):
        h = torch.cat([x.view(-1, self.img_size * self.img_size), self.class_embedding(labels)], dim=1)
        h = self.relu(self.fc1(h))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        z = torch.cat([z, self.class_embedding(labels)], dim=1)
        h = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h))  # Ensure output is in [0, 1] by using sigmoid


    def forward(self, x, labels):
        mu, logvar = self.encode(x.view(-1, 1 * 28 * 28), labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar