import torch
import torchvision.models.alexnet as alexnet
import torch.utils.data as Data
import torch.nn as nn
from torchvision import datasets,models,transforms
from torch.autograd import Variable
import os
from PIL import Image
import matplotlib.image as mpimg
import numpy as np

model = alexnet()
num_ftrs = model.classifier._modules['6'].in_features
model.fc = nn.Linear(num_ftrs,3)

model.load_state_dict(torch.load('ill_alexnet.pkl'))

im = Image.open('test/test1.jpg')
im = im.resize((224,224))
im_array = np.array(im,dtype=np.float32)/255.0
im_array = im_array.transpose((2,0,1))#调整通道顺序
tensor = torch.from_numpy(im_array)


input = Variable(tensor.unsqueeze(0))
print(input.size())
outputs = model(input)
_,preds = torch.max(outputs.data,1)
print(preds)