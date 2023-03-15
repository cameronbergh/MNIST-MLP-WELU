import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

class SELUMnistNet(nn.Module):
    def __init__(self):
        super(SELUMnistNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.selu1 = nn.SELU()
        self.fc2 = nn.Linear(256, 64)
        self.selu2 = nn.SELU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.selu1(self.fc1(x))
        x = self.selu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize wandb
wandb.init(project="mnist-selu", entity="impudentstrumpet")
config = wandb.config

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SELUMnistNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Log the model architecture to wandb
wandb.watch(model)

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            # Log metrics to wandb
            wandb.log({"Train Loss": loss.item(),
                       "Train Step": epoch * len(train_loader) + batch_idx})

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    # Log metrics to wandb
    wandb.log({"Test Loss": test_loss,
               "Test Accuracy": accuracy,
               "Epoch": epoch})


# Training loop
num_epochs = 20
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, criterion, optimizer, epoch)
    test(model, device, test_loader, criterion, epoch)
