import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable as Var

train_path = "processed_data/train.npz"
test_path = "processed_data/test.npz"
# train_path = "processed_data/raw_train.npz"
# test_path = "processed_data/raw_test.npz"
batch_size = 256
epoches = 30
learning_rate = 0.0001

class network1(nn.Module):
    def __init__(self):
        super(network1, self).__init__()
        self.linear1 = nn.Linear(453, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(256, 69)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        out = self.linear4(x)
        return out

class network2(nn.Module):
    def __init__(self):
        super(network2, self).__init__()
        self.linear1 = nn.Linear(22283, 8192)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(8192, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(256, 69)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        out = self.linear4(x)
        return out

class DualDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_dataloader():
    train_data = np.load(train_path)
    test_data = np.load(test_path)

    train_targets = train_data["arr_0"]
    train_size = train_targets.shape[0]
    train_targets = torch.LongTensor(train_targets)
    train_set = []
    for i in range(train_size):
        train_set.append(train_data["arr_%d" % (i + 1)])


    test_targets = test_data["arr_0"]
    test_size = test_targets.shape[0]
    test_targets = torch.LongTensor(test_targets)
    test_set = []
    for i in range(test_size):
        test_set.append(test_data["arr_%d" % (i + 1)])

    train_set = torch.Tensor(train_set)
    test_set = torch.Tensor(test_set)

    train_dataset = DualDataset(train_set, train_targets)
    test_dataset = DualDataset(test_set, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("dataloader prepared")
    return train_loader, test_loader


def train(train_loader,test_loader):
    model = network1().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoches):
        print("epoch: ", epoch, " / ", epoches)

        correct = 0
        losses = []
        train_num = len(train_loader.sampler)

        for data, label in train_loader:
            data = data.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(
                output, label, size_average=False
            )
            loss.backward()
            optimizer.step()
            losses.append(loss)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max logit
            correct += int(
                pred.eq(label.view_as(pred)).sum()
            )  # add to running total of hits
        print("Loss: ",float(torch.mean(torch.Tensor(losses))))
        print(
            "Train Accuracy: {}/{} ({:.2f}%)".format(
                correct, int(train_num), 100.0 * float(correct / train_num)
            )
        )

    test_correct = 0
    test_num = len(test_loader.sampler)
    for data, label in test_loader:
        data = data.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        output = model(data)

        pred = output.max(1, keepdim=True)[1]
        test_correct += int(
            pred.eq(label.view_as(pred)).sum()
        )
    print(
        "Test Accuracy: {}/{} ({:.2f}%)".format(
            test_correct, test_num, 100.0 * test_correct / test_num
        )
    )

    return test_correct/test_num


def main():
    train_loader, test_loader = prepare_dataloader()
    max_accuracy = -1
    for i in range(1):
        print("Try ",i)
        accuracy = train(train_loader,test_loader)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
    print("Max Accuracy:", max_accuracy)

main()