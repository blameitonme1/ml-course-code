import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(10816, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

loss = nn.CrossEntropyLoss()
model = cnn()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

def training_step(model):
    pbar = tqdm(train_dataloader)
    for (X, y) in pbar:
        y_pred = model(X)
        loss_value = loss(y_pred, y)
        optimizer.zero_grad()
        pbar.set_description(f"loss: {loss_value:.4f}")  # 更新进度条描述信息
        loss_value.backward()
        optimizer.step()
    return model

def training_loop(model, epochs=10):
    for epoch in range(epochs):
        model = training_step(model)
    return model

def test_step(model):
    total = 0
    acc = 0
    for (X, y) in tqdm(test_dataloader):
        # calculate a batch accuracy
        y_pred = model(X)
        acc += (torch.argmax(y_pred, 1) == y).sum().item()
        total += y.size(0)
    return (acc / total)

model = training_loop(model=model, epochs=1)
print(test_step(model))