import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy
EPOCH = 100
BATCH_SIZE = 50
LR = 0.001



if torch.cuda.is_available():
    print('支持GPU')
else:
    print('不支持')

train_data = torchvision.datasets.ImageFolder(
    'wheat_data',
    transform=torchvision.transforms.ToTensor()
)

train_loader = Data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)
print('data_loader:%s'%len(train_loader))

test_data = torchvision.datasets.ImageFolder(
    'wheat_test',
    transform=torchvision.transforms.ToTensor()
)
test_loader = Data.DataLoader(test_data,batch_size=len(test_data),shuffle=True)

test_x = None
test_y = None
for x,y in test_loader:
    test_x = Variable(x).type(torch.FloatTensor).cuda()
    test_y = Variable(y).cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*56*56,3)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x) # 32,56,56
        x = x.view(x.size(0),-1) # 保留维度0 吧后面的 32,56,56 展开成32*56*56
        output = self.out(x)
        return output

cnn = CNN()
cnn.cuda()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x).cuda()
        print(b_x.size())
        b_y = Variable(y).cuda()
        output = cnn(b_x)

        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        optimizer.step()

        if step % 50 == 0:
            print(test_x.size())
            test_ouput = cnn(test_x)
            pred_y = torch.max(test_ouput,1)[1].cuda().data.squeeze()
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0])

test_ouput = cnn(test_x[:10])
pred_y = torch.max(test_ouput,1)[1].cuda().data.squeeze()
print(pred_y,'prediction number')
print(test_y[:10],'real number')