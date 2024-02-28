#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:53:18 2019

@author: tong
"""
import os
from os import listdir
import numpy as np
import random
import shutil

fname = listdir('./CAL500_wav')
train = random.sample(fname,k=400)
a = [f for f in fname if f not in train]
valid = random.sample(a, k=50)
test = [f for f in a if f not in valid]

# train
trainpath = "./CAL500Data/train"
validpath = "./CAL500Data/valid"
testpath = "./CAL500Data/test"
for root, dirs, files in os.walk("./CAL500_wav"):
	for name in files:
		if name in train:
			shutil.copyfile(root + "/" + name, trainpath + "/" + name)
		if name in valid:
			shutil.copyfile(root + "/" + name, validpath + "/" + name)
		if name in test:
			shutil.copyfile(root + "/" + name, testpath + "/" + name)

trainlabelpath = "./Label/train"
validlabelpath = "./Label/valid"
testlabelpath = "./Label/test"

for root, dirs, files in os.walk("./SegLabelHard"):
	for name in files:
		if name[0:-4]+".wav" in train:
			shutil.copyfile(root + "/" + name, trainlabelpath + "/" + name)
		if name[0:-4]+".wav" in valid:
			shutil.copyfile(root + "/" + name, validlabelpath + "/" + name)
		if name[0:-4]+".wav" in test:
			shutil.copyfile(root + "/" + name, testlabelpath + "/" + name)