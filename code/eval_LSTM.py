from __future__ import print_function, division, absolute_import, unicode_literals
import six
import argparse
import os
import numpy as np
import Model_noLSTM
import Data
import torch
import util
import sys

blockSize=2048
hopSize=1024
batchSize=1

# Load test set
datasetTest = Data.CAL500Data(wavpath = PATH_DATASET, labelpath = PATH_LABEL, split = 'test', nfft=blockSize, hopsize=hopSize, repetition = 1)
dataloaderTest = torch.utils.data.DataLoader(datasetTest, batch_size = batchSize, shuffle=False, num_workers = 0)

# Call the model
model = Model_noLSTM.Model()
checkpoint = torch.load("savedModel_RNN_best.pt", map_location=lambda storage, loc:storage)

model.eval()

TP = np.zeros(50,18)
FP = np.zeros(50,18)
FN = np.zeros(50,18)
TN = np.zeros(50,18)
seqLength = np.zeros(50)
nFrame = 0

for idx, (x,y) in emumerate(dataloaderTest):
	x = x.permute(0,2,3,4,1)
	y = y.permute(0,2,1)
	seqLen = x.shape[4]
	seqLength[idx] = seqLen
	currentBatchSize = x.shape[0]
	state = Model.initstate(currentBatchSize)
	state = (state[0], state[1])
	for f in range(seqLen):
		output,state = model(x[:,:,:,:,f],state)
		for l in range(output.shape[1]):
			if y[0,l] = 1:
				if output[0,l] = 1:
					TP[idx,l] += 1
				else: FN[idx,l] += 1
			if y[0,l] != 1:
				if output[0,l] = 1:
					FP[idx,l] += 1
				else: TN[idx,l] += 1
		
MAP = 
print(TP,FP,FN,TN)

	

