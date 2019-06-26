'''
Utility functions
'''
import numpy as np
import scipy.io.wavfile as sio
import csv

def vectorize(audio,rate,window_size,group_size = 3, full = False,training = False, labels = np.array([]), serial = False):
    '''
    windows and vectorizes input data, then combined together input vectors into groups determined by group_size while retaining the correct ordering. The labels for the combined data is 1 if any of the labels within those windows were 1. If full = True the remaining tail of data will be kept after windowing is performed. If training = true, label data needs to be input. if serial = true, the output will be every sequenctial grouping of the vectors. 
    
    inputs: 
    audio = audio input as an array
    rate = sampling rate of audio
    window_size = size of sample window in seconds
    group_size = amount of vectors to combine into groups
    full = bool, if True final window at the tail end of smaller size will no be cutt off
    training = bool, if True, then a labels input is also expected
    Labels = vector of labels for classification task
    serial = bool, if True, increases the number of training samples by using every sequential combination of the vectors
    
     I.e If serial is true and group_size = 3, the output will be every sequential combination of 3 vectors.
    
    output (with serial = False) = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                                   [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    output (with serial = True) = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                                  [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                                  [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
                                  [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
                                  
    
    outputs:
    data = matrix of windowed ffts for samples of signal determined by window_size. Each fft is split into 5 bins whithin the voice range(80-3000) which are averaged to create 5d vectors for each window. These vectors are then bunched into groups determined by group_size
    
    labels = if training = True, downsampled labels to have the same number of entries as data
    '''
    
    #find number of windows that can fully fit
    window_size = int(rate*window_size) #convert from seconds to samples
    num_windows = len(audio) // window_size #find number of windows that can fully fit
    
    #split audio and label into windows
    if full == True and training == True: #Split audio and labels and keep the tail portion
        audio_tail = audio[int(window_size*num_windows):]
        labels_tail = labels[int(window_size*num_windows):] 
        
        audio = np.split(audio[0:int(window_size*num_windows)] , num_windows)
        labels = np.split(labels[0:int(window_size*num_windows)] , num_windows)
        
        audio.append(audio_tail)
        labels.append(labels_tail)
        
        labels = [1 if 1 in i else 0 for i in labels] #reform labels array into a vector with a 1 for each entry window that has a 1
        labels = np.array(labels).reshape(len(labels),1)
        
    elif full == False and training == True: #Split Audio and labels and do not keep tail portion
        audio = np.split(audio[0:int(window_size*num_windows)] ,  num_windows)
        labels = np.split(labels[0:int(window_size*num_windows)] , num_windows)
        
        labels = [1 if 1 in i else 0 for i in labels] #reform labels array into a vector with a 1 for each entry window that has a 1
        labels = np.array(labels).reshape(len(labels),1)
        
    
    elif full == True and training == False: #Split Audio and keep tail portion
        audio_tail = audio[int(window_size*num_windows):]
        audio = np.split(audio[0:int(window_size*num_windows)] , num_windows)
        audio.append(audio_tail)
        
    elif full == False and training == False: #Split audio and do not keep tail portion    
        audio = np.split(audio[0:int(window_size*num_windows)] , num_windows)
    
    
    
    #perform fft and averaging on each window
    data = np.zeros((len(audio),5)) #create empty data array
    for i in range(len(audio)): 
        fft = abs(np.fft.fft(audio[i])) #fft on windowed sample
        freq = np.fft.fftfreq(fft.shape[-1] ,1/rate) #determine bin size for fft
        scale = freq[1]
        
        
        fft = fft[int(80/scale):int(3000/scale)] #cut fft to range between 80 and 3000hz
        num_windows = 5
        window_size = len(fft) // num_windows
        
        fft = np.split(fft[0:int(num_windows*window_size)],num_windows) #split fft into 5 windows of equal size
        
        new_entry = np.round(np.array( [np.mean(i) for i in fft] ) ) #round each window to get a 5d vector
        data[i,:] = new_entry
        
        
    #group vector data together
    total = data.shape[0] - group_size + 1
    num_groups = data.shape[0] // group_size
    
    if serial == True and training == True: #group data and labels in serial fashion
        temp_data = data[0:group_size,:].reshape(1,group_size*data.shape[1])
        temp_labels = labels[0:group_size].reshape(1,group_size)
        
        for i in range(total - 1):
            temp_data = np.vstack((temp_data, data[1+i:group_size+1+i,:].reshape(1,group_size*data.shape[1])  ))
            temp_labels = np.vstack((temp_labels, labels[1+i:group_size+1+i].reshape(1,group_size)))
            
        labels = np.array([1 if 1 in i else 0 for i in temp_labels]).reshape(total,1)
        data = temp_data
    elif serial == False and training == True: #group data and labels
        data = data[0:num_groups*group_size].reshape(num_groups,group_size*data.shape[1])
        labels = labels[0:num_groups*group_size].reshape(num_groups,group_size)
        labels = np.array([1 if 1 in i else 0 for i in labels]).reshape(num_groups,1)
        
    elif serial == True and training == False: #group data in serial fashion
        temp_data = data[0:group_size,:].reshape(1,group_size*data.shape[1])
        
        for i in range(total - 1):
            temp_data = np.vstack((temp_data, data[1+i:group_size+1+i,:].reshape(1,group_size*data.shape[1])  ))
        data = temp_data
     
    elif serial == False and training == False: #group data
        data = data[0:num_groups*group_size].reshape(num_groups,group_size*data.shape[1])
        
        
    
    data = data / np.amax(data) # normalize by maximm value
    if training == True:
        return data, labels
    return data


def vectorize_teensy(path,group_size=3):
    '''
    Averages, and combines together fft data from teensy serial monitor. 
    
    inputs: 
    path = path to csv file that contains 68 bin fft(2-70) from teensy devkit sampled at 86hz with bins 43 hz in between 
    group_size = amount of vectors to combing into groups
    
    
    outputs:
    data = matrix of grouped, windowed ffts for samples of signal determined by window_size and group_size. Each fft is split into 5 bins whithin the voice range(80-3000) which are averaged to create 5d vectors for each window. These vectors are then clumped into groups of size group_size
    
    '''
    #convert csv file to numpy array
    fft = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            fft.append(row)
    fft.pop(-1) #remove first and last entries to ensure stability
    fft.pop(0)
    fft = np.array(fft)
    
    
    
    data = np.zeros((fft.shape[0],5))
    for i in range(fft.shape[0]):    
        
        next_fft = fft[i] #goes through each fft
        next_fft = np.split(next_fft[0:65],5) #splits into 5 even windows
        next_entry = np.array([np.mean(i) for i in next_fft]) #averages the windows
        data[i] = next_entry
        
    #group the data together
    total = data.shape[0] - group_size + 1
    num_groups = data.shape[0] // group_size
    
    data = data[0:num_groups*group_size].reshape(num_groups,group_size*data.shape[1])
    #data = data / np.amax(data) # normalize by maximm value
   
    return data
    

