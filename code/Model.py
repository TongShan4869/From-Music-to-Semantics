#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:52:01 2019

@author: tong
"""

from __future__ import print_function, division, absolute_import, unicode_literals
import six
import os
import numpy as np
import Data
import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import argparse
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class ModelLSTM(torch.nn.Module):
	def __init__(self):
		super(ModelLSTM, self).__init__()
		# Input layer input size of 128x128x1
		self.conv1 = nn.Conv2d(1,10,kernel_size=3,padding=1,stride=1)
		self.relu1 = nn.ReLU()
		
		self.conv2 = nn.Conv2d(10,20,kernel_size=3,padding=1,stride=1)
		self.relu2 = nn.ReLU()
		
		self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
		
		self.conv3 = nn.Conv2d(20,40,kernel_size=3, padding=1,stride=1)
		self.relu3 = nn.ReLU()

		self.conv4 = nn.Conv2d(40,80,kernel_size=3, padding=1,stride=1)
		self.relu4 = nn.ReLU()

		self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
		
		self.fc1 = nn.Linear(32*32*80, 1000)
		self.relu4 = nn.ReLU()
		
		self.lstm = nn.LSTMCell(1000,500)
		
		self.fc2 = nn.Linear(1500,600)
		self.relu5 = nn.ReLU()
		
		self.out = nn.Linear(600,18)
		self.sigmoid = nn.Sigmoid()
		
		self.initParams()
		
	def initParams(self):
		for param in self.parameters():
			if len(param.shape)>1:
				torch.nn.init.xavier_normal_(param)

	def initState(self, batchSize):
		hx = torch.randn(batchSize,500)
		cx = torch.randn(batchSize,500)
		state = (hx,cx)
		return state
	
	def convlayer(self,x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pool1(x)
		x = self.conv3(x)
		x = self.relu3(x)
		x = self.conv4(x)
		x = self.relu4(x)
		y = self.pool2(x)
		return y

	def forward(self,x,state):
		x = self.convlayer(x)
		x = x.view(-1, 32*32*80)
		x = self.fc1(x)
		x = self.relu4(x)
		state = self.lstm(x,state)
		r = state[0]
		x = torch.cat((r,x),1)
		x = self.fc2(x)
		x = self.relu5(x)
		x = self.out(x)
		x = self.sigmoid(x)
		return x, state

def validate(model, dataloaderValid):
	with torch.no_grad():
		model.eval()
	#Each time fetch a batch of samples from the dataloader
		iterator = iter(dataloaderValid)
		epochValidloss = 0
		with trange(len(dataloaderValid)) as t:
			for idx in t:
				#Each time fetch a batch of samples from the dataloader
				x,y = next(iterator)

	######################################################################################
	# Implement here your validation loop. It should be similar to your train loop
	# without the backpropagation steps
	######################################################################################
	#read the input and the fitting target into the device
				x = x.to(device)
				x = x.permute(0,2,3,4,1)
				y = y.to(device)
				y = y.permute(0,2,1)
				y = y.long()

				seqLen = x.shape[4]
				currentBatchSize = x.shape[0]

				#output = torch.zeros((currentBatchSize,outputSize))


				# init state
				state = model.initState(currentBatchSize)
				state = (state[0].to(device), state[1].to(device))

				validloss = 0

				for f in range(seqLen):
					output,state = model(x[:,:,:,:,f],state)
					# loss per frame
					validloss += lossfunc(output, y[:,:,f])

				# Compute loss per batch
				epochValidloss += validloss / seqLen
		aveValidloss = epochValidloss / len(dataloaderValid)

	# this is used to set a description in the tqdm progress bar 
	t.set_description(f"validation loss {aveValidloss}")
	model.train()
	return aveValidloss

if __name__ == "__main__":
	######################################################################################
	# Load Args and Params
	######################################################################################
	parser = argparse.ArgumentParser(description='Train Arguments')
	parser.add_argument("--blockSize", type=int, default = 2048)
	parser.add_argument('--hopSize', type=int, default = 1024)
	# how many audio files to process fetched at each time, modify it if OOM error
	parser.add_argument('--batchSize', type=int, default = 1)
	# set the learning rate, default value is 0.0001
	parser.add_argument('--lr', type=float, default=1e-5)
	# Path to the dataset, modify it accordingly
	parser.add_argument('--dataset', type=str, default = './CAL500Data')
	# Path to the label, modify it accordingly
	parser.add_argument('--label', type=str, default = './Label')
	# set --load to 1, if you want to restore weights from a previous trained model
	parser.add_argument('--load', type=int, default = 0)
	# path of the checkpoint that you want to restore
	parser.add_argument('--checkpoint', type=str, default = 'savedModel_RNN_LSTM_best.pt')
	
	parser.add_argument('--seed', type=int, default = 555)
	args = parser.parse_args()

	# Random seeds, for reproducibility
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	blockSize = args.blockSize
	hopSize = args.hopSize
	PATH_DATASET = args.dataset
	PATH_LABEL = args.label
	batchSize= args.batchSize
	minValLoss = np.inf

	# Dataloader
	datasetTrain = Data.CAL500Data(wavpath = PATH_DATASET, labelpath = PATH_LABEL, split = 'train', nfft=blockSize, hopsize=hopSize, repetition = 1)
	datasetValid = Data.CAL500Data(wavpath = PATH_DATASET, labelpath = PATH_LABEL, split = 'valid', nfft=blockSize, hopsize=hopSize, repetition = 1)

	#initialize the data loader
	#num_workers means how many workers are used to prefetch the data, reduce num_workers if OOM error
	dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size = batchSize, shuffle=False, num_workers = 2)
	dataloaderValid = torch.utils.data.DataLoader(datasetValid, batch_size = batchSize, shuffle=False, num_workers = 0)

	#initialize the Model
	model = ModelLSTM()

	# if you want to restore your previous saved model, set --load argument to 1
	if args.load == 1:
		checkpoint = torch.load(args.checkpoint)
		minValLoss = checkpoint['minValLoss']
		model.load_state_dict(checkpoint['state_dict'])

	#determine if cuda is available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	#initialize the optimizer for paramters
	optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
	lossfunc = nn.MultiLabelMarginLoss()

	model.train(mode=True)

	#################################### 
	#The main loop of training
	#################################### 
	for epoc in range(100):
		iterator = iter(dataloaderTrain)
		with trange(len(dataloaderTrain)) as t:
			for idx in t:
				#Each time fetch a batch of samples from the dataloader
				x, y = next(iterator) # x is the input, y is the target
				#the progress of training in the current epoch

				#Remember to clear the accumulated gradient each time you perfrom optimizer.step()
				optimizer.zero_grad()

				#read the input and the fitting target into the device
				x = x.to(device)
				x = x.permute(0,2,3,4,1)
				y = y.to(device)
				y = y.permute(0,2,1)
				y = y.long()

				seqLen = x.shape[4]
				currentBatchSize = x.shape[0]

				#output = torch.zeros((currentBatchSize,outputSize))


				# init state
				state = model.initState(currentBatchSize)
				state = (state[0].to(device), state[1].to(device))

				#################################
				# Fill the rest of the code here#
				#################################
				loss = 0
				for f in range(seqLen):

					output,state = model(x[:,:,:,:,f],state)

					# loss per frame
					loss += lossfunc(output,y[:,:,f])


				# Compute loss per batch
				trainloss = loss / seqLen

				# Backward
				trainloss.backward()

				# Update weight
				optimizer.step()

				# this is used to set a description in the tqdm progress bar 
				t.set_description(f"epoc : {epoc}, loss {trainloss}")
				#save the model

		# create a checkpoint of the current state of training
		checkpoint = {
					'state_dict': model.state_dict(),
					'minValLoss': minValLoss,
					}
		# save the last checkpoint
		torch.save(checkpoint, 'savedModel_RNN_LSTM_last.pt')

		#### Calculate validation loss
		valLoss = validate(model, dataloaderValid)
		print(f"validation Loss = {valLoss:.4f}")

		if valLoss < minValLoss:
			minValLoss = valLoss
			# then save checkpoint
			checkpoint = {
					'state_dict': model.state_dict(),
					'minValLoss': minValLoss,
					}
			torch.save(checkpoint, 'savedModel_RNN_LSTM_best.pt')
		torch.cuda.empty_cache()
