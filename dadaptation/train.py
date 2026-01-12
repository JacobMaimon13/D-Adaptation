# filename: train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import argparse
import os


try:
    from dadaptation.dadapt_asgd import DAdaptASGD
    print("Successfully imported DAdaptASGD.")
except ImportError:
    try:
        from dadapt_asgd import DAdaptASGD
    except ImportError:
        print("Error: Could not find dadapt_asgd.py in 'dadaptation/' folder.")
        exit(1)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Loading CIFAR-10 Data...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    net = SimpleCNN().to(device)

    print("Initializing D-Adaptation ASGD...")
    optimizer = DAdaptASGD(net.parameters(), lr=1.0, weight_decay=args.wd, log_every=100)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Loss={running_loss/len(trainloader):.4f}, Acc={acc:.2f}%")

    print("Training Finished.")
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(net.state_dict(), 'checkpoints/model_asgd.pth')
    print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--wd', type=float, default=0.0)
    args = parser.parse_args()
    train(args)
