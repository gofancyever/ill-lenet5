
from torchvision import datasets,transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
inputDir = 'wheat_test_or'
outputDir = 'wheat_test'
from PIL import Image
import scipy.misc



class Rotate(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, param):
        self.param = param


    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        return img.transpose(self.param)



transforms_data = [
    transforms.Compose([
        transforms.Scale(255),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.Scale(255),
        transforms.CenterCrop(224),
        Rotate(Image.ROTATE_90),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.Scale(255),
        transforms.CenterCrop(224),
        Rotate(Image.ROTATE_180),
        transforms.ToTensor()
    ]),
    transforms.Compose([
        transforms.Scale(255),
        transforms.CenterCrop(224),
        Rotate(Image.ROTATE_270),
        transforms.ToTensor()
    ])
]




# Visualize a few images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    plt.show()
    if title is not None:
        plt.title(title)
def saveImg(inp,file):
    inp = inp.numpy().transpose((1, 2, 0))
    print(inp.shape)
    scipy.misc.imsave(file, inp)



for idx,tansform in enumerate(transforms_data):
    data_sets = datasets.ImageFolder(inputDir,transform=transforms_data[idx])
    data_loader = torch.utils.data.DataLoader(data_sets,)
    classes = data_sets.classes
    print('data num:',len(data_sets))
    print('class:',classes)
    for index,(data,label) in enumerate(data_loader):
        dir = classes[label[0]]
        dir = '%s/%s/' % (outputDir, dir)
        isExists = os.path.exists(dir)
        if not isExists:
            # 如果不存在则创建目录
            print(dir + ' 创建成功')
            # 创建目录操作函数
            os.makedirs(dir)
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print(dir + ' 目录已存在')

            dirname = '%s/%s_%s.jpg' % (dir,str(index),str(idx))
            saveImg(data[0],dirname)