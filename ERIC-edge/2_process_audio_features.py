# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000

import subprocess
import getopt
import sys
import os
import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
import datetime as dt
import time
import sklearn
from sklearn import preprocessing

import cProfile as profile

#-----------------------------------------------------------------
def get_start_time(wav_fn):
  # print('wav_fn = ', wav_fn)
  if wav_fn[:8] == 'doorbell': # 'doorbell-20210416-200901-1618621741.wav' 2021 radu home video file name
    time = np.datetime64(dt.datetime.strptime(wav_fn[9:24],"%Y%m%d-%H%M%S"))

  elif wav_fn[:5] == 'Front': # 'Front-20221017-035243.mp4'  2022 radu new home data
    time = np.datetime64(dt.datetime.strptime(wav_fn[6:23],"%Y-%m-%d-%H%M%S"))

  elif wav_fn[:4] == 'Back': # 'Back-20221017-035243.mp4'  2022 radu new home data
    time = np.datetime64(dt.datetime.strptime(wav_fn[5:22],"%Y-%m-%d-%H%M%S"))

  else: 
    # for annotated filename Heavy_2021-04-23T18-43-30.wav
    #fn = wav_fn.split('_')[1]
    
    # '2021-04-24T07-00-01.wav' tamu west campus video filename
    fn = wav_fn
    time = np.datetime64(dt.datetime.strptime(fn,"%Y-%m-%dT%H-%M-%S.wav"))
    
  return time

#-----------------------------------------------------------------
def amplitude_envelope(signal, frame_size, hop_length):
    """
    Fancier Python code to calculate the amplitude envelope of a signal with a given frame size.
    """
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_length)])

#-----------------------------------------------------------------
def cal_audio_fea(x, sr, FRAME_LENGTH, HOP_LENGTH, fea_path, wav_fn, is_plot):
  
  cal_start = time.time()
  #-----------zero crossing rate
  zero_crossing_rate = librosa.feature.zero_crossing_rate(x, frame_length = FRAME_LENGTH, hop_length = HOP_LENGTH)[0]

  #-----------RMSE
  #rmse = librosa.feature.rms(x, S=None, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
  # more accurate using S
  s, phase = librosa.magphase(librosa.stft(x, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))
  rmse_s = librosa.feature.rms(S=s, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

  # same timestamps for all features if using the same frame length and hop length
  t_fea = librosa.frames_to_time(range(len(rmse_s)), sr=sr, hop_length=HOP_LENGTH)

  #------------spectral centroids
  spectral_centroids_s = librosa.feature.spectral_centroid(S=s, n_fft = FRAME_LENGTH, hop_length = HOP_LENGTH, sr=sr)[0]
  #spectral_centroids = librosa.feature.spectral_centroid(x, n_fft = FRAME_LENGTH, hop_length = HOP_LENGTH, sr=sr)[0]

  #------------spectral rolloff
  spectral_rolloff_s = librosa.feature.spectral_rolloff(S=s, n_fft = FRAME_LENGTH, hop_length = HOP_LENGTH, sr=sr)[0]
  #spectral_rolloff = librosa.feature.spectral_rolloff(x, n_fft = FRAME_LENGTH, hop_length = HOP_LENGTH, sr=sr)[0]

  #------------amplitude envelope
  ae = amplitude_envelope(x, frame_size=FRAME_LENGTH, hop_length=HOP_LENGTH)

  #------------MFCCs
  mel_s = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
  mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_s), sr=sr, n_mfcc=20, dct_type=2)
  mfccs = mfccs.astype(float)
  mfccs = sklearn.preprocessing.scale(mfccs, axis=1) # scale the MFCCs
  
  cut = 0
  if len(ae)<len(rmse_s): # in case len(ae)<len(rmse_s) by 1
    cut = len(ae)
  else:
    cut = len(rmse_s)
  
  # get the time label column
  start_time = get_start_time(wav_fn)
  date=np.empty(len(t_fea), dtype='datetime64[ns]')
  for i in range(len(t_fea)):
    date[i]= start_time + np.timedelta64(i,'s') # each data point is 1 sec

  af = pd.DataFrame()
  af['DateTime'] = date[:cut]

  # build the af dataframe
  af['ZeroCrossingRate'] = zero_crossing_rate[:cut]
  af['RMSE'] = rmse_s[:cut]
  af['SpectralCentroids'] = spectral_centroids_s[:cut]
  af['SpectralRolloff'] = spectral_rolloff_s[:cut]
  af['AmplitudeEnvelope'] = ae[:cut]
  
  # MFCCs
  for i in range(mfccs.shape[0]):
    name = 'MFCCS_'+ str(i+1)
    af[name] = mfccs[i,:cut]

  af_name = wav_fn.split('.')[0] 
  af.to_csv(fea_path + '/' + af_name + '_audio.csv', index = False)
  cal_end = time.time()
  cal_time = cal_end - cal_start
  print(', cal_time =', round(cal_time,2), 's', end = ' ')

  # ------------------------- plot the features
  if is_plot:
    plot_start = time.time()
    fig, axs = plt.subplots(5, 1,figsize=(14,12))
    fig.suptitle(wav_fn)

    xmax = len(rmse_s) # each frame = 1 s
    r=0
    t_x = np.arange(len(x))/sr
    axs[r].plot(t_x,x,label='Amplitude')
    axs[r].set_ylim(-0.01, 0.01)
    axs[r].plot(t_fea[:cut], ae,label='Amplitude envelope', color ='red',alpha=0.5)
    #axs[r].set_xlabel('Time (sec)')
    axs[r].set_ylabel('Amplitude (normalized)')
    axs[r].set_xlim(0, xmax)
    axs[r].legend(loc="upper right")
    r+=1

    #axs[r].plot(t_fea,rmse, label = 'RMSE',color ='r')
    axs[r].plot(t_fea, rmse_s, label = 'RMSE')
    #axs[r].set_xlabel('Time (sec)')
    axs[r].set_ylabel('RMSE')
    axs[r].set_ylim(-0.001, 0.001)
    axs[r].set_xlim(0, xmax)
    axs[r].legend(loc="upper right")
    r+=1

    axs[r].plot(t_fea, zero_crossing_rate, label='Zero Crossing Rate')
    #axs[r].set_xlabel('Time (sec)')
    axs[r].set_ylabel('Zero Crossing Rate')
    axs[r].set_xlim(0, xmax)
    axs[r].legend(loc="upper right")
    r+=1

    axs[r].plot(t_fea, spectral_centroids_s, label='Spectral Centroids')
    #axs[r].set_xlabel('Time (sec)')
    #axs[r].set_ylabel('Frequency (Hz)')
    #r+=1

    axs[r].plot(t_fea, spectral_rolloff_s, label='Spectral Rolloff', color ='red', alpha = 0.5)
    axs[r].set_xlabel('Time (sec)')
    axs[r].set_ylabel('Frequency (Hz)')
    axs[r].set_xlim(0, xmax)
    axs[r].legend(loc="upper right")
    r+=1
    
    plt.subplot(5, 1, 5)
    librosa.display.specshow(mfccs, sr=sr, x_axis='frames')
    plt.colorbar(format='%+02.0f dB', orientation = 'horizontal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    plt.savefig(fea_path + '/' + af_name + '_Fea.jpg')
    plt.close()
    
    plot_end = time.time()
    plot_time = plot_end - plot_start
    print(', plot_time =', round(plot_time, 2), 's', end = ' ')

#-----------------------------------------------------------------  
def get_audiofea(wav_path, wav_fn, fea_path, is_plot):
  
  wav_f = wav_path + '/'+ wav_fn
  # load the wav file
  #overwrite the default sampling rate of 22kHz to use the origional sampling rate in the file
  x, sr = librosa.load(wav_f, sr=None)
  # librosa will automatically normalize the data to [-1,1]
  print('len(x)=',len(x), end = ' ')
  print('sampling_rate=', sr, 'Hz', end = ' ')
  print('duration=', round(len(x)/sr,2),'s', end =' ')

  if len(x) == 0: # empty audio track
    print("\n!!! Empty audio track in", wav_f, ", skipped!\n")
    return

  # calculate the feature per second
  FRAME_LENGTH = sr
  #FRAME_LENGTH = int(sr/30) # 30 frames per second
  HOP_LENGTH = FRAME_LENGTH # no overlapping
  
  cal_audio_fea(x, sr, FRAME_LENGTH, HOP_LENGTH, fea_path, wav_fn, is_plot)

#-----------------------------------------------------------------
def main():
  """
  $ python3 process_audio_features.py -w -d ./2021-09-03/
  $ python3 process_audio_features.py -w -d ./ -p
  $ python ../../Code/process_audio_features.py -w -s 15 -p

  -w: extract all wav files, otherwise use available wav files
  -d: video_dir
  -s: folder_start, from 1
  -a: wav_start, from 1
  -p: use all the subfolders under current video_dir as work_dir
  """

  start_time = time.time()
  # default
  video_dir = './'
  extract_wav = False
  parent_dir = False
  folder_start = 1
  wav_start = 1
  # is_plot = True
  is_plot = False

  try:
      options, remainder = getopt.getopt(sys.argv[1:], 'wd:s:a:p')
  except getopt.GetoptError as err:
      print("getopt error: %s" % (str(err)))
      return

  for opt, earg in options:
    if (opt == '-w'):
      extract_wav = True
    if (opt == '-d'):
      video_dir = earg
    if (opt == '-s'):
      folder_start = int(earg)
    if (opt == '-a'):
      wav_start = int(earg)
    if (opt == '-p'):
      parent_dir = True

  print('video_dir =', video_dir)
  print('folder_start =', folder_start)

  if parent_dir:
    subfolders = os.listdir(video_dir)
    # subfolders = glob.glob("*/")
    print('# of subfolders in current parent dir:', len(subfolders))
    subfolders.sort()
    for i in range(len(subfolders)):
      print(i+1, subfolders[i])
  else:
    subfolders = ['']


  # make wav folder
  # wav_path = work_dir + 'wav'
  wav_path = video_dir + 'wav/'
  print("wav_path =", wav_path)
  if not os.path.isdir(wav_path):
    os.mkdir(wav_path)
    print("Folder {} created".format(wav_path))        
  else:
    print("Folder {} already existed".format(wav_path))
  
  # make wav/AudioFeatures folder
  # fea_path = wav_path + '/AudioFeatures'
  fea_path = video_dir + 'AudioFeatures/'
  print("fea_path =", fea_path)
  if not os.path.isdir(fea_path):
    os.mkdir(fea_path)
    print("Folder {} created".format(fea_path))        
  else:
    print("Folder {} already existed".format(fea_path))

  # create the log file
  log_filepath = fea_path + '/AudioProcessor.log'
  fwlog = open(log_filepath,'a')

  #-------------------extract wav files-------------------
  for folder in subfolders[folder_start-1 : ]:
    work_dir = video_dir + folder
    print('\nwork_dir =', work_dir)
    # list all .mp4 files
    files = os.listdir(work_dir)
    fn_list = [i for i in files if i.endswith('.mp4')]
    print("Total # of Video Files: ", len(fn_list))
    fn_list.sort()
  
    # extract all wav files from videos
    if extract_wav:
      extract_s = time.time()
      for f in fn_list:
        wav_fn = f.split('.')[0]+'.wav'
        print('extracting ' + wav_fn + '......')

        # sync the audio time in case of missing audios
        command = 'ffmpeg -i '+ work_dir + '/' + f + ' -async 1 '+ wav_path + '/' + wav_fn
        subprocess.call(command, shell=True, stdout=fwlog, stderr=fwlog)
        fwlog.flush()
      extract_e = time.time()
      print('Extract wav files costs {}s'.format(round(extract_e-extract_s, 2)))
  
  #-------------------calculate audio features-------------------
  wav_files = os.listdir(wav_path)
  wav_fn_list = [i for i in wav_files if i.endswith('.wav')]
  print("Total # of wav Files: ", len(wav_fn_list))
  wav_fn_list.sort()

  i = wav_start
  for wav_fn in wav_fn_list[wav_start-1:]:
    print('get_audiofea():' + wav_path + '/' + wav_fn, end=' ')
    print('%d/%d' % (i, len(wav_fn_list)), end = ' ')

    cal_start = time.time()
    get_audiofea(wav_path, wav_fn, fea_path, is_plot) #+++++++

    print('total_time', round(time.time() - cal_start, 2), "s", end= '\n')
    i += 1
  
  fwlog.close()
  print('Done!')
  total_time = time.time() - start_time
  print('============== Total time:', round(total_time/60, 2), 'mins')
     
#-----------------------------------------------------------------
if __name__ == "__main__":
  # profile.runctx('main()', globals(),locals(), sort='cumtime')
  main()