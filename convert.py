# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:35:31 2019

@author: bdgecyt
"""

from PIL import Image
import os
from os import listdir
import matplotlib as plt



for file in listdir("C:\\Users\\bdgecyt\\Desktop\\dataset\\Household Shelter"):
#    print(files)
    if file.endswith(".jpg"):
        print(file)
        
#        str.endswith(suffix)
        im = Image.open("C:\\Users\\bdgecyt\\Desktop\\dataset\\Household Shelter\\" + file)
        print(file[:-3] + "png")
        im.save("C:\\Users\\bdgecyt\\Desktop\\dataset\\Household Shelter\\" + file[:-3] +"png")
        
        
im = Image.open("C:\\Users\\bdgecyt\\Desktop\\dataset\\Household Shelter\\test2_84.png")

im.size

img = plt.pyplot.imread("C:\\Users\\bdgecyt\\Desktop\\dataset\\Household Shelter\\test2_84.png")
img = plt.pyplot.imread("C:\\Users\\bdgecyt\\Desktop\\dataset\\Household Shelter\\271_25-1_00_23_41_0_2.png")

plt.pyplot.imread('Household Shelter/271_25-1_00_23_41_0_2.png')



image = plt.imread('Household Shelter/271_25-1_00_23_41_0_2.png')

matplotlib.pyplot.imread