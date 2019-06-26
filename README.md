# voice-wearable-volunteer-project
For the purpose of voice activity detection and voice distinction tasks, this package uses a trained multilayer perceptron to identify voice features.  Pre-trained weights are provided, but the network can be easily changed and retrained with any data that the user would like to use. 

 
CLASSIFICATION:


Input: The input can be in 1 of 2 forms. Either  raw audio from wav files with the associated sample rate, or the path to a csv file containing the copy and pasted serial monitor output of the modified fft example from the teensy audio library.

Preprocessing: For preprocessing, use either the vectorize or the vectorize teensy functions within the utils.py file. 

The vectorize function splits the audio into windows (the user can specify the window size/sample_rate). An fft is performed on each window. The range is reduced to only include frequencies within the audio range(80-3000hz). These remaining bins are then pooled evenly into 5 bins by averaging. This results in an array of 5d vectors where each vector represents the  downsampled fft for a window of audio data. These vectors are then combined into groups of 3 to form size 15 vectors(group_size can be changed). All values are then normalized by the maximum value. 

Example: an audiosample 9 seconds long with windows of 0.5 seconds  and group size of 3 would result in an array of size 6x15. 

The vectorize_teensy function takes as input the path to the csv file generated from copying the serial monitor output from arduino. It cuts off the first and last entries (sometimes the serial monitor will not display these as full arrays). Afterwards each fft vector is averaged into 5 bins and every 3 vectors is grouped together to form groups of 15 (group size can be changed).
 

Network: 
The preprocessed data can be input to multilayer perceptron for classification. The network is a 5 layered multilayer perceptron with input, output, and 3 hidden layers Implemented with keras on tensorflow. The current input is size 15 vectors and the output is a soft max layer that shows the probabilities for 2 separate results.  Any of the layers can be easily altered or changed by the user, however this would require retraining of the model.
 



TRAINING:

For preprocessing use the appropriate vectorize function and switch the positional argument training = TRUE., and include the labels. This will perform the same steps as noted above, but each set of 15 vectors will be matched with the appropriate label.

From there, the model can be trained and an accuracy can be outputted. The user can choose to save their weights if they wish





Hardware:
Teensy dev kit

SGTL-5000 DSP chip

Software:

Python Libraries:

Tensorflow

Numpy

Scipy

Os

Time

csv

Arduino Libraries:
Teensy Audio Library

CODE:
Utils.py -Contains functions for converting raw audio data into windowed, downsampled fast fourier transform vectors
cmi_rnn_vad.ipnyb -Ipython notebook with examples for executing Voice Activity Classification on saved wav files or fft input from teensy devkit. 
Recorded Data folder
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
