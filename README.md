# voice-wearable-volunteer-project
For the purpose of voice activity detection and voice distinction tasks, this package uses a Recurrent Neural Network to identify voice features.  Pre-trained weights are provided, but the network can easily be changed, retrained, or remodeled with any data that the user would like to use. (Compatible with teensy 3.6 with Audio Adapter Board)


# TRAINING:

Both training and classification are done with rnn_vad.pynb. Example audio data within the data/audio directory is used to train the model. Any wav files with appropriate labels can be used for training.

Input: The input to the model is raw audio from wav files with the associated sample rate and labels. These are processed by the vectorize function inside of utils.py. Vectorize takes the audio and labels, splits the audio into windows. On each of these windows an fft is performed. The window size determines the bin size for the fft (bin_size = sample_rate/window_size). The output will be vectors of size 15 representing the averaged fft for 3 audio windows.


From there, the model can be trained and an accuracy can be outputted. The user can choose to save their weights if they wish



# CLASSIFICATION:


Very similar to Training, but does not require labeled data. Process the input data using vectorize.py, load the saved model and predict. 




# Hardware:
Teensy dev kit

SGTL-5000 DSP chip

# Software:

Python Libraries:

Tensorflow

Numpy

Scipy

Os

Time

csv

Arduino Libraries:
Teensy Audio Library

# CODE:
Utils.py -Contains functions for converting raw audio data into windowed, downsampled fast fourier transform vectors

rnn_vad.ipnyb -Ipython notebook with examples for executing Voice Activity Classification on saved wav files or fft input from teensy devkit. 

Data folder

--audio
Bus_stop.mat

Bus_stop.wav

Construction_site.mat

Construction_site.wav

Park.mat

Park.wav

Room.mat

room.wav


Weights folder

my_model

checkpoint

my_model.data-00000-of-00001

my_model.index
