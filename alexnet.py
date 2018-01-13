import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import os
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data
import time
from torch.optim import lr_scheduler
from logger import Logger


logger = Logger('./logs')

plt.ion()


# load data

data_transforms = {
    'wheat_data':transforms.Compose([
        transforms.ToTensor()
    ]),
    'wheat_test':transforms.Compose([
        transforms.ToTensor()
    ])
}

data_dir = 'data'
image_datasets = { x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['wheat_data','wheat_test'] }
dataloaders = { x: Data.DataLoader(image_datasets[x],batch_size=4,shuffle=True,num_workers=4) for x in ['wheat_data','wheat_test'] }
dataset_sizes = { x:len(image_datasets[x]) for x in ['wheat_data','wheat_test'] }
class_names = image_datasets['wheat_data'].classes
use_gpu = torch.cuda.is_available()

def imshow(inp,title=None):
    '''显示Tensor类型的图片'''
    inp = inp.numpy().transpose((1,2,0))#调整通道顺序
    plt.imshow(inp)
    if title is not  None:
        plt.title(title)
    plt.show()

#一批训练集
inputs,classes = next(iter(dataloaders['wheat_data']))
#对图片制作网格
out = torchvision.utils.make_grid(inputs)
imshow(out,title=[class_names[x] for x in classes])



def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()
    best_acc = 0.0
    best_model_wts = model.state_dict()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch,num_epochs-1))
        print('-'*10)

        for phase in ['wheat_data','wheat_test']:
            if phase == 'wheat_data':
                scheduler.step()
                model.train(True)# 将魔心设置为训练模式
            else:
                model.train(False)
            running_loss = 0.0
            running_corrects = 0

            #迭代数据
            for i,data in enumerate(dataloaders[phase]):
                #得到输入数据
                inputs,labels = data
                #包装
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs,labels = Variable(inputs),Variable(labels)
                # 梯度归零
                print(inputs.size())
                optimizer.zero_grad()

                #向前传播
                outputs = model(inputs)
                _,preds = torch.max(outputs.data,1)
                loss = criterion(outputs,labels)
                #反向传播+参数优化 如果是处于训练时期
                if phase == 'wheat_data':
                    loss.backward()
                    optimizer.step()
                #对每次迭代的loss 和accuracy 求和
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            #统计每轮平均loss 和 accuracy
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} loss:{:.4f} acc:{:.4f}'.format(phase,epoch_loss,epoch_acc))
            # ========================= Log ======================
            step = epoch
            # (1) Log the scalar values
            info = {'loss': epoch_loss, 'accuracy': epoch_acc}

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)





            # 保存最好的模型
            if phase == 'wheat_test' and epoch_acc>best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()

    time_elapsed = time.time()-since
    print('training complete in {:.0f}m {:.0f}s'.format(time_elapsed/60,time_elapsed%60))
    print('best wheat_test acc:{:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model,num_images=6):

    images_so_far = 0
    fig = plt.figure()
    for i,data in enumerate(dataloaders['wheat_test']):
        inputs,labels = data

        if use_gpu:
            inputs,labels = Variable(inputs.cuda()),Variable(labels.cuda())
        else:
            inputs,labels = Variable(inputs),Variable(labels)
        outputs = model(inputs)
        _,preds = torch.max(outputs.data,1)

        tag = [class_names[pred] for pred in preds]
        logger.image_summary(tag,inputs.cpu().data,i)
        # for j in range(inputs.size()[0]):
        #     images_so_far += 1
        #     ax = plt.subplot(num_images/2,2,images_so_far)
        #     ax.axis('off')
        #     ax.set_title('predicted:{}'.format(class_names[preds[j]]))
        #     plt.plot(inputs.cpu().data[j])
        #     if images_so_far == num_images:
        #         return
    # fig.show()
model_ft = models.alexnet(pretrained=True)
print(model_ft)

num_ftrs = model_ft.classifier._modules['6'].in_features
model_ft.fc = nn.Linear(num_ftrs,3)
if use_gpu:
    model_ft = model_ft.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(),lr=0.001,momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=7,gamma=0.1)
model_ft = train_model(model_ft,criterion,optimizer_ft,exp_lr_scheduler,num_epochs=24)
visualize_model(model_ft)
torch.save(model_ft.state_dict(), 'ill_alexnet.pkl')
