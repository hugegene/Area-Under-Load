# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:24:58 2019

@author: bdgecyt
"""

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torch.optim as optim
import torch
from torchvision import transforms, utils
torch.cuda.set_device(-1)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import json
from torchvision import transforms
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
from functools import reduce
import operator
import math

class DopeNetwork(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=4,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage
        
        if pretrained is False:
            print("Training network without imagenet weights.")
        else:
            print("Training network pretrained on imagenet.")

        vgg_full = models.vgg19(pretrained=pretrained).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
        i_layer = 23
        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        self.vgg.add_module(str(i_layer+2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap,
                                             numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap,
                                             numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap,
                                             numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap,
                                             numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap,
                                             numBeliefMap, False)

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m6_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
    def forward(self, x):
        '''Runs inference on the neural network'''

        out1 = self.vgg(x)
#         print(out1.size())

        out1_2 = self.m1_2(out1)
#         out1_1 = self.m1_1(out1)
#         print("stage1")
#         print(out1_2.size())
#         print(out1_1.size())

        if self.stop_at_stage == 1:
            return [out1_2],
#                    [out1_1]

        out2 = torch.cat([out1_2, out1], 1)
        out2_2 = self.m2_2(out2)
#         out2_1 = self.m2_1(out2)
        
#         print("stage2")
#         print(out2_2.size())
#         print(out2_1.size())

        if self.stop_at_stage == 2:
            return [out1_2, out2_2],
#                    [out1_1, out2_1]

        out3 = torch.cat([out2_2, out1], 1)
        out3_2 = self.m3_2(out3)
#         out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2],
#                    [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out1], 1)
        out4_2 = self.m4_2(out4)
#         out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2],
#                    [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out1], 1)
        out5_2 = self.m5_2(out5)
#         out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2],
#                    [out1_1, out2_1, out3_1, out4_1, out5_1]

        out6 = torch.cat([out5_2, out1], 1)
        out6_2 = self.m6_2(out6)
#         out6_1 = self.m6_1(out6)
#         print("stage6")
#         print(out6_2.size())
#         print(out6_1.size())

        return out1_2, out2_2, out3_2, out4_2, out5_2, out6_2
#                [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]
                        
    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.add_module("0",
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding)
                        )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(str(i),
                             nn.Conv2d(
                                 mid_channels,
                                 mid_channels,
                                 kernel_size=kernel,
                                 stride=1,
                                 padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model

def padding(img):
  z = np.zeros((700, 700, 3), dtype=img.dtype)
  z[:img.shape[0], :img.shape[1], :] = img
  return z

def crop(img):
  return img[:400,:400,:]


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized    
    
if __name__ == '__main__' :
    print("a")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DopeNetwork(pretrained=False).to(device)
    net = torch.nn.DataParallel(net).to(device)
    print(net)
    net.load_state_dict(torch.load("C:\\Users\\bdgecyt\\Desktop\\poseparameters.pth", map_location=torch.device('cpu')))
    
    load = image_resize(load, width = 400)
    resize = load.shape
    load = padding(load)
    load = load[:400,:400,:]
    
    
    data_transform = transforms.Compose([transforms.ToTensor()])
    
    inp = data_transform(load)
    inp = inp.view([1, 3, 400, 400])
 
    out = net(inp)[0].data.numpy().reshape(4, 50, 50)
    
    predictedcoor = []
    for i in out:
#        print(i.shape)
        argmax = np.unravel_index(i.argmax(), i.shape)
        print(argmax)
        argmax = np.array(argmax)/50
        print(argmax)
        predictedcoor += [argmax]
    
    glo
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), glo), [len(glo)] * 2))
    coords = sorted(glo, key=lambda coord: (200 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    
    for i in predictedcoor:
         print(i[0], i[1])
#        print(i[0]*400, i[1]*400)
#        print(i[1]*400)
         cv2.circle(load, tuple([int(i[1]*resize[1]), int(i[0]*resize[0])]) ,3,(255,0,0),-1)
        
    cv2.imshow("image", load)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    