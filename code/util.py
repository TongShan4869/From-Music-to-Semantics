#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:02:41 2019

@author: tong
"""

import numpy as np
import os
import scipy.io.wavfile as wav
import librosa
from six.moves import cPickle 
import matplotlib.pyplot as plt
import pandas as pd


def create_melspec(wavfile,n_fft,hop_length):
	(rate, sample) = wav.read(wavfile)
	sample = sample.astype(float)
	melspec = librosa.feature.melspectrogram(sample,float(rate), n_fft=n_fft, hop_length=hop_length)
	nframe = melspec.shape[1]
	samp_len = len(sample)
	return melspec, nframe, rate, samp_len

def create_label_mat(labelpath, rate, nframe, samp_len):
	label = pd.read_csv(labelpath)
	label_s = label.sort_values(by='Start_Time').reset_index()
	label_s = label_s.drop(columns = 'index')
	#y_val = np.zeros((label_s.shape[1]-2,nframe))
	y_val = np.zeros((18,nframe)) #18 emotion tags
	start_ind = 0
	for i in range(len(label_s)):
		if i!= len(label_s)-1:
			end_time = label_s["End_Time"][i]
			end_ind = np.round((end_time*rate)/samp_len*nframe)
			for f in range(int(start_ind),int(end_ind)):
				y_val[:,f] = label_s.iloc[i].as_matrix()[2:20]
			start_ind = end_ind+1
		else: 
			end_ind = nframe
			for f in range(int(start_ind),int(end_ind)):
				y_val[:,f] = label_s.iloc[i].as_matrix()[2:20]
	y_val = y_val.astype(bool)
	return y_val

def long_term_frame(melspec, y_val, nframe, winsize=128, hopsize=64):  #compute new long term frame of 100 frames (about 4.5 seconds)
	n_long_frame = np.floor((nframe-winsize)/hopsize).astype(int)
	x = [] 
	x_mel = np.zeros((1, melspec.shape[0],winsize))
	y = []
	y_lab = np.zeros(y_val.shape[0])
	for i in range(n_long_frame):
		x_mel = melspec[:,i*hopsize:(i*hopsize + winsize)]
		x_mel = np.expand_dims(x_mel, axis=0)
		x.append(x_mel)
		y_lab = np.mean(y_val[:,i*hopsize:(i*hopsize + winsize)],axis=1)
		y_lab = np.where(y_lab > 0.5, 1, 0)
		y.append(y_lab)
	return x, y
		

if __name__ == "__main__":
	(melspec, nframe, rate,samp_len) = create_melspec("./CAL500Data/train/2pac-trapped.wav", 2048, 1024)
	y_val = create_label_mat("./Label/train/2pac-trapped.csv",rate,nframe,samp_len)
	(x,y) = long_term_frame(melspec,y_val,nframe,winsize=128,hopsize=64)
