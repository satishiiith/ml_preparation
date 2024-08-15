import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        distances = (output2 - output1).pow(2).sum(1)  # Squared Euclidean distance
        losses = 0.5 * (target.float() * distances +
                        (1 - target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean()

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, input1, input2):
        output1 = self.cnn(input1)
        output2 = self.cnn(input2)
        return output1, output2

def generate_pairs(batch):
    # Randomly transform images within each batch to create pairs
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    batch = torch.stack([transform(x) for x in batch])
    return batch, batch  # Return each image and its transformed version as pairs

# Load data
dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Initialize network and loss
net = SiameseNetwork()
criterion = ContrastiveLoss(margin=1.0)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Train for 10 epochs
    for images, _ in dataloader:
        img1, img2 = generate_pairs(images)
        output1, output2 = net(img1, img2)
        loss = criterion(output1, output2, torch.ones(img1.size(0)))  # All pairs are positive
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
