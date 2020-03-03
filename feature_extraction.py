from glob import glob
import numpy as np
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path

def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = './urban/train/' + name + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def create_spectrogram_test(filename,name):
    plt.interactive(True)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = Path('./urban/test/' + name + '.png')
    fig.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


Data_dir=np.array(glob("./Datasets/8Kprocessed/to4sec/*"))

i=0
for file in Data_dir[i:i+2000]:
    #Define the filename as is, "name" refers to the PNG, and is split off into the number itself.
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram(filename,name)
gc.collect()

i=2000
for file in Data_dir[i:i+2000]:
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram(filename,name)
gc.collect()

i=4000
for file in Data_dir[i:]:
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram(filename,name)
gc.collect()

Test_dir=np.array(glob("./Datasets/8Kprocessed/4test/*"))
i=0
for file in Test_dir[i:i+1500]:
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram_test(filename,name)
gc.collect()

i=1500
for file in Test_dir[i:]:
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram_test(filename,name)
gc.collect()
