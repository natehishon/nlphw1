# q-5
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

input_size = 784
num_classes = 10
learning_rate = 0.01
batch_size = 30
epochs = 30

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FC_NN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(FC_NN, self).__init__()
        self.input = nn.Linear(input_size, 256)
        self.hidden = nn.Linear(256, 64)
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.sigmoid(self.input(x))
        x = F.sigmoid(self.hidden(x))
        x = self.output(x)
        return x


def process(outputs):
    encoded = [onehot(output.item()) for output in outputs]
    return torch.tensor(encoded)


def onehot(output):
    x = [0.0 for _ in range(num_classes)]
    x[output] = 1.0
    return x


model = FC_NN(input_size=input_size, num_classes=num_classes).to(device)

criterion = nn.MSELoss(reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for batch_idx in enumerate(train_loader):
        x, (y, output) = batch_idx
        y = y.reshape(batch_size, -1)
        outputs = model(y)
        loss = criterion(outputs, process(output))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_accuracy(data, model, data_type):
    correct = 0
    total = 0
    model.eval()

    for x, y in data:
        x = x.to(device=device)
        y = y.to(device=device)
        x = x.reshape(x.shape[0], -1)

        scores = model(x)
        z, pred = scores.max(1)
        correct += (pred == y).sum()
        total += pred.size(0)

    accuracy = correct / total
    print(data_type + " accuracy: {}".format(accuracy))

    model.train()


evaluate_accuracy(train_loader, model, "Training Set")
evaluate_accuracy(test_loader, model, "Testing Set")
