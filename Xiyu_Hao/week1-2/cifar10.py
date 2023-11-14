import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import cv2


class myModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,5,1,2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024,64)
        self.linear2 = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.linear1(x)

        x = self.linear2(x)
        x = self.softmax(x)

        return x

# load dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 32
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,num_workers=1)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,num_workers=1)

myModel = myModel()

trainsize = len(trainset)
testsize = len(testset)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(myModel.parameters(), lr=0.01)

epochs = 50
for epoch in range(epochs):
    print("No.{}/{}".format(epoch + 1, epochs))

    train_total_loss = 0.0
    test_total_loss = 0.0

    train_total_acc = 0.0
    test_total_acc = 0.0


    # training
    for data in trainloader:
        inputs,labels = data

        optimizer.zero_grad()

        outputs = myModel(inputs)

        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        _,index = torch.max(outputs, 1)
        acc = torch.sum(index == labels).item()

        train_total_loss += loss.item()
        train_total_acc += acc

    with torch.no_grad():
        for data in testloader:
            inputs,labels = data

            outputs = myModel(inputs)
            loss = loss_func(outputs, labels)
            _,index = torch.max(outputs, 1)
            acc = torch.sum(index == labels).item()
            test_total_loss += loss.item()
            test_total_acc += acc

    print("train loss:{}, train acc:{}, test loss:{}, test acc:{}".format(train_total_loss,train_total_acc/trainsize,test_total_loss,test_total_acc/testsize))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# test
folder_path = './testimages'

files = os.listdir(folder_path)
images_files = [os.path.join(folder_path, f) for f in files]

for img in images_files:
    image = cv2.imread(img)
    cv2.imshow('image', image)
    image = cv2.resize(image, (32, 32))
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = transform(image)
    torch.reshape(image, (1, 3, 32, 32))
    output = myModel(image)

    value,index = torch.max(output, 1)
    pre_val = classes[index]

    print('Prediction:{}, Index:{}, Result:{}'.format(value.item(), index.item(), pre_val))
    cv2.waitKey(0)