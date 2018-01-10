import torch
import torchvision
from torchvision import transforms,utils
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

EPOCH = 100
BATCH_SIZE = 50
LR = 0.001


img_data = torchvision.datasets.ImageFolder(
    'wheat',
    transform=transforms.Compose(
        [
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
    )
)

img_test = torchvision.datasets.ImageFolder(
    'wheat_test',
    transform=transforms.Compose(
        [
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]
    )
)

print('数据总数:%s'%len(img_data))
data_loader = torch.utils.data.DataLoader(img_data,batch_size = BATCH_SIZE,shuffle=True)
print('data_loader:%s'%len(data_loader))

print('测试数据总数:%s'%len(img_test))
test_loader = torch.utils.data.DataLoader(img_test,shuffle=True)
print('test_loader:%s'%len(test_loader))

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()


# for step, (batch_x, batch_y) in enumerate(data_loader):  # 每一步 loader 释放一小批数据用来学习
#     print(step)
#     print(batch_x)
#     print(batch_y)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5) # output 220 * 220 *6
        self.pool = nn.MaxPool2d(2,2) #output 110 * 110 * 6
        self.conv2 = nn.Conv2d(6,16,5) # output 106 * 106 * 16 -> pool : 53 * 53 * 16
        self.fc1 = nn.Linear(16*53*53,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,3)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=LR)

for epoch in range(EPOCH):
    runing_loss = 0.0
    for i,data in enumerate(data_loader,0):
        inputs,labels = data
        inputs,labels = Variable(inputs),Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        runing_loss += loss.data[0]
        if i % 5 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])
print('finish')

for i,data in enumerate(test_loader,0):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    test_output = net(inputs)
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print('==============')
    print(pred_y, 'prediction number')
    print(labels, 'real number')
    print('==============')