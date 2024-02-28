#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:35:07 2019

@author: tong
"""


import os
from pydub import AudioSegment


source_path = "./CAL500_32kps"
dest_path = "./CAL500_wav"

for dirName, subdirList, fileList in os.walk(source_path):
    for fname in fileList:
        file_name = dirName + "/" + fname
        out_name = fname[0:-3] + "wav"
        file = AudioSegment.from_mp3(file_name)
        file.export(dest_path + "/" + out_name, format="wav")


# flac to wav
wav_path = "/home/tong/AMPLab/MusicABR/pilot_2_music_types"
files_path = wav_path + "/Funk"

for dirName, subdirList, fileList in os.walk(files_path):
    for fname in fileList:
        file_name = dirName + "/" + fname
        out_name = dirName + "/" + fname[0:-4] + "wav"
        file = AudioSegment.from_file(file_name)
        file.export(out_name, format="wav")