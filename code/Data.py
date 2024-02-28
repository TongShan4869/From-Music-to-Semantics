#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:47:25 2019

@author: tong
"""

from __future__ import print_function, division, absolute_import, unicode_literals
import six
import os
import numpy as np
import util
import sys
import torch
import torchvision.transforms as transforms 
import torch.utils.data



class CAL500Data(torch.utils.data.Dataset):
	def __init__(self, wavpath = './CAL500Data', labelpath='./Label', split = 'train', nfft=2048, hopsize=1024, repetition = 1):
		wave_path = wavpath+"/"+split
		label_path = labelpath+"/"+split
		wave_paths = []
		label_paths = []
		f_names = []
		for f in os.listdir(wave_path):
			filepath = wave_path + "/" + f
			labelpath = label_path + "/" + f[0:-3] + "csv"
			wave_paths.append(filepath)
			label_paths.append(labelpath)
			f_names.append(f)
			
		self.wave_paths = wave_paths*repetition
		self.label_paths = label_paths*repetition
		self.f_names = f_names
		self.nfft = nfft
		self.hopsize = hopsize
		
	def __len__(self):
		return len(self.wave_paths)
	
	def __getitem__(self, idx):
		file_path = self.wave_paths[idx]
		file_name = self.f_names[idx]
		label_path = self.label_paths[idx]
		(melspec, nframe, rate, samp_len) = util.create_melspec(file_path, self.nfft, self.hopsize)
		y_val = util.create_label_mat(label_path,rate,nframe,samp_len)
		(x,y) = util.long_term_frame(melspec,y_val,nframe,winsize=128,hopsize=64)
		x = torch.FloatTensor(x)
		y = torch.FloatTensor(y)
		return x,y
	 
if __name__ == "__main__":
	torch.manual_seed = 555
	nfft = 2048
	hopsize = 1024
	dataset = CAL500Data(wavpath = './CAL500Data', labelpath='./Label', split = 'valid', nfft=nfft, hopsize=hopsize, repetition = 1)
	print(len(dataset))
	dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False, num_workers = 0)
	dataloaderIt = iter(dataloader)
	for idx, (x,y,file_name) in enumerate(dataloader):
		print((idx,file_name,x.shape,y.shape))
