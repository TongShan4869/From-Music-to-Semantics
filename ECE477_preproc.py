# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:02:28 2019

@author: TShan
"""
import numpy as np
import os
import scipy.io.wavfile as wav
import librosa
import pydub
from pydub import AudioSegment
from six.moves import cPickle 
import python_speech_features as features
import matplotlib.pyplot as plt
import pandas as pd

## ==================
##  MP3 TO WAVE
## ==================
#mp3_path = "C:/Users/TShan/Downloads/CAL500_32kps/CAL500_32kps"
wav_path = "C:/Users/TShan/Downloads/CAL500_32kps/CAL500_wave"
#for root, dirs, file in os.walk(mp3_path):
#	for f in file:
#		sound = AudioSegment.from_mp3(os.path.join(mp3_path,f))
#		sound.export(wav_path + "/" + f[0:-3] + "wav",format='wav')
label_path = "C:/Users/TShan/Downloads/CAL500_32kps/SegLabelHard"
target_path = wav_path + '/preprocessed'


## ==================
##  SPLIT
## ==================
##### SCRIPT VARIABLES #####
train_size 	= 400
val_size 	= 100
test_size 	= 100



##### SCRIPT META VARIABLES #####
VERBOSE = True
DEBUG 	= True
debug_size = 5
	# Convert only a reduced dataset
visualize = False







def create_melspec(wavfile):
	(rate, sample) = wav.read(wavfile)
	sample = sample.astype(float)
	melspec = librosa.feature.melspectrogram(sample,float(rate), n_fft=2048, hop_length=1024)
	return melspec, melspec.shape[1]


#Debug example
rate, sample = wav.read(wav_path+"/2pac-trapped.wav")
(out,nframe) = create_melspec(wav_path+"/2pac-trapped.wav")
label = pd.read_csv(label_path + "/2pac-trapped.csv")
label_s = label.sort_values(by='Start_Time').reset_index()
label_s = label_s.drop(columns = 'index')
y_val = np.zeros((label_s.shape[1]-2,nframe))
start_ind = 0
for i in range(len(label_s)):
	if i!= len(label_s)-1:
		end_time = label_s["End_Time"][i]
		end_ind = np.round((end_time*rate)/len(sample)*nframe)
		for f in range(int(start_ind),int(end_ind)):
			y_val[:,f] = label_s.iloc[i].as_matrix()[2:69]
		start_ind = end_ind+1
	else: 
		end_ind = nframe
		for f in range(int(start_ind),int(end_ind)):
			y_val[:,f] = label_s.iloc[i].as_matrix()[2:69]
y_val = y_val.astype(bool)
	



#Plot
plt.figure(0)
x_axis 	= np.linspace(0, len(sample)/rate, num=len(sample))
ax1 = plt.subplot(3,1,1)
plt.plot(x_axis, sample)
ax1.set_xlim([0, len(sample)/rate])

plt.ylabel('Original wave data')
plt.tick_params(
		axis='both',       	# changes apply to the axis
		which='both',      	# both major and minor ticks are affected
		bottom='off',      	# ticks along the bottom 
		top='off',         	# ticks along the top 
		right='off',      	# ticks along the right 
		left='off',      	# ticks along the left
		labelbottom='off', 	# labels along the bottom
		labelleft='off')		# labels along the top
plt.tight_layout()


plt.subplot(3,1,2)
plt.imshow(np.log(out), interpolation='nearest', aspect='auto', cmap='plasma',origin='lower')
# plt.axis('off')
# plt.title('Preprocessed data')

plt.ylabel('Melspectrogram data')
plt.tick_params(
		axis='both',       	# changes apply to the axis
		which='both',      	# both major and minor ticks are affected
		bottom='off',      	# ticks along the bottom 
		top='off',         	# ticks along the top 
		right='off',      	# ticks along the right 
		left='off',      	# ticks along the left
		labelbottom='off', 	# labels along the bottom
		labelleft='off')		# labels along the top
plt.tight_layout()
	

plt.subplot(3,1,3)
plt.imshow(y_val, interpolation='nearest', aspect='auto', cmap='plasma',origin='lower')
# plt.axis('off')
# plt.title('Preprocessed data')

plt.ylabel('Labeled data')
plt.tick_params(
		axis='both',       	# changes apply to the axis
		which='both',      	# both major and minor ticks are affected
		bottom='off',      	# ticks along the bottom 
		top='off',         	# ticks along the top 
		right='off',      	# ticks along the right 
		left='off',      	# ticks along the left
		labelbottom='off', 	# labels along the bottom
		labelleft='off')		# labels along the top
plt.tight_layout()








def preprocess_dataset(source_path, VERBOSE=False, visualize=False):
	"""Preprocess data, ignoring compressed files and files starting with 'SA'"""
	i = 0
	X = []
	Y = []
	fig = []
	num_plot = 5

	for dirName, subdirList, fileList in os.walk(source_path):
		for fname in fileList:
			if fname.endswith('.csv') : 
				continue

			label_fname = dirName + '/' + fname
			wav_fname = dirName + '/' + fname[0:-4] + '.wav'

			total_duration = get_total_duration(phn_fname)
			fr = open(label)
			if visualize:
				curr_fig      = plt.figure(i)
				frame_rate, wave_sig = wav.read(wav_fname)
#				wav_file 	= wave.open(wav_fname, 'r')
#				signal 		= wav_file.readframes(-1)
#				signal 		= np.fromstring(signal, 'Int16')
#				frame_rate 	= wav_file.getframerate()
#				if wav_file.getnchannels() == 2:
#					print('ONLY MONO FILES')

				x_axis 	= np.linspace(0, len(wave_sig)/frame_rate, num=len(wave_sig))
				ax1 	= plt.subplot(num_plot,1,1)
				# plt.title('Original wave data')
				plt.plot(x_axis, wave_sig)
				ax1.set_xlim([0, len(wave_sig)/frame_rate])

				plt.ylabel('Original wave data')
				plt.tick_params(
					axis='both',       	# changes apply to the axis
					which='both',      	# both major and minor ticks are affected
					bottom='off',      	# ticks along the bottom 
					top='off',         	# ticks along the top 
					right='off',      	# ticks along the right 
					left='off',      	# ticks along the left
					labelbottom='off', 	# labels along the bottom
					labelleft='off')		# labels along the top
				plt.tight_layout()

				# plt.gca().axes.get_xaxis().set_visible(False)
				f,t,X_spec = scipy.signal.spectrogram(wave_sig, fs=frame_rate, window=('tukey', 0.25), 
                                               nperseg=None, noverlap=None, nfft=None, 
                                               detrend='constant', return_onesided=True, 
                                               scaling='density', axis=-1, mode='psd')
            


			X_val, total_frames = create_mfcc('DUMMY', wav_fname)
			total_frames = int(total_frames)

			X.append(X_val)
			if visualize:

				plt.subplot(num_plot,1,3)
				plt.imshow(X_spec, interpolation='nearest', aspect='auto', cmap='plasma')
				plt.ylabel('Original Spectrogram')
				plt.tick_params(
					axis='both',       	# changes apply to the axis
					which='both',      	# both major and minor ticks are affected
					bottom='off',      	# ticks along the bottom 
					top='off',         	# ticks along the top 
					right='off',      	# ticks along the right 
					left='off',      	# ticks along the left
					labelbottom='off', 	# labels along the bottom
					labelleft='off')		# labels along the top
				plt.tight_layout()

				plt.subplot(num_plot,1,4)
				plt.imshow(X_val.T, interpolation='nearest', aspect='auto', cmap='plasma')
				# plt.axis('off')
				# plt.title('Preprocessed data')

				plt.ylabel('Preprocessed data')
				plt.tick_params(
					axis='both',       	# changes apply to the axis
					which='both',      	# both major and minor ticks are affected
					bottom='off',      	# ticks along the bottom 
					top='off',         	# ticks along the top 
					right='off',      	# ticks along the right 
					left='off',      	# ticks along the left
					labelbottom='off', 	# labels along the bottom
					labelleft='off')		# labels along the top
				plt.tight_layout()


			y_val = np.zeros(total_frames) - 1
			start_ind = 0
			for line in fr:
				[start_time, end_time, phoneme] = line.rstrip('\n').split()
				start_time = int(start_time)
				end_time = int(end_time)

				phoneme_num = find_phoneme(phoneme)
				end_ind = np.round((end_time)/total_duration*total_frames)
				y_val[int(start_ind):int(end_ind)] = phoneme_num

				start_ind = end_ind
			fr.close()

			if -1 in y_val:
				print('WARNING: -1 detected in TARGET')
				print(y_val)

			Y.append(y_val.astype('int32'))
			if visualize:
				plt.subplot(num_plot,1,2)
				plt.imshow((y_val.T, ), aspect='auto', cmap='Pastel2')

				plt.ylabel('Lables')
				plt.tick_params(
					axis='both',       	# changes apply to the axis
					which='both',      	# both major and minor ticks are affected
					bottom='off',      	# ticks along the bottom 
					top='off',         	# ticks along the top 
					right='off',      	# ticks along the right 
					left='off',      	# ticks along the left
					labelbottom='off', 	# labels along the bottom
					labelleft='off')		# labels along the top

			if visualize:
				plt.subplots_adjust(hspace=0.01)
				plt.tight_layout()
				# plt.show()
				fig.append(curr_fig)

			i+=1
			if VERBOSE:
				print()
				print('({}) create_target_vector: {}'.format(i, phn_fname[:-4]))
				print('type(X_val): \t\t {}'.format(type(X_val)))
				print('X_val.shape: \t\t {}'.format(X_val.shape))
				print('type(X_val[0][0]):\t {}'.format(type(X_val[0][0])))
			else:
				print(i, end=' ', flush=True)
			if i >= debug_size and DEBUG:
				break

		if i >= debug_size and DEBUG:
			break
	print()
	return X, Y, fig