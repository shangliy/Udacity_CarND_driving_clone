#!/usr/locals/python
#author:Shanglin Yang(kudoysl@gmail.com)

'''
This script is to transform csv/json input data to pickle structure, meanwhile,
The data being smoothed using 'pyasl.smooth'
'''

import json
import csv
import pickle

import matplotlib.pyplot as plt

import numpy as np
from PyAstronomy import pyasl
from scipy.misc import imread, imresize
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter



def generate_arrays_from_csv(file_name):
    '''
    Generator for csv file.    
    This works for using directly the csv file from simulator
    '''
    while 1:
        zero_count = 0
        with open(file_name) as f:
            cf = csv.reader(f)
            for row in cf:
                
                center_img_path = row[0]
                            
                x = np.zeros(shape=(1,IMG_HEIGHT,IMG_WEIGHT,3))
                x[0] = imresize(imread(center_img_path),(IMG_HEIGHT,IMG_WEIGHT))

                y = np.zeros(shape=(1,))
                y[0] = float(row[3])
                if abs(y[0])<0.1:
                    zero_count += 1 
                    if zero_count < 1:
                        continue
                    else:
                        zero_count = 0
                        yield center_img_path,y    
                else:
                    yield center_img_path,y

def generate_arrays_from_json(file_name):
    
    '''
    Generator for json file.    
    This works for using data refined by human
    '''
    while 1:
        zero_count = 0
        with open(file_name) as f:
            train_data = json.load(f)
        for data in train_data:
        
            image_path = data['path']
            angle = data['angle'] 
            center_img_path = image_path
            x = np.zeros(shape=(1,IMG_HEIGHT,IMG_WEIGHT,3))
            x[0] = imresize(imread(center_img_path),(IMG_HEIGHT,IMG_WEIGHT))
            y = np.zeros(shape=(1,))
            y[0] = float(angle)
            yield center_img_path,y

Traing_sample_num = 4890+2013+126+146+507 #The number of required training data
Show_num = 100 # The number to show
X_path = [] # Image path
y = np.zeros((Traing_sample_num,))

# Training data Generator

#y_o = generate_arrays_from_file('./evaluation/driving_log.csv')
#y_o = generate_arrays_from_file('./training_data/driving_log.csv')
#y_o = generate_arrays_from_file('./train_4/driving_log.csv')
#y_o = generate_arrays_from_file('./final_eval/driving_log.csv')
y_o = generate_arrays_from_json('./train_4.json')  

# Whether to show sample of training data
img2show = True
train_data = {}

for i in range(Traing_sample_num):
    print (i)
    path,ang = next(y_o)
    y[i] = ang
    X_path.append(path)

train_data['img_path'] = X_path

#Smooth data
sm2 = pyasl.smooth(y, 11, 'hamming')
train_data['angle'] = sm2

# Save the pickle file
with open('train_4_new.p', 'wb') as handle:
  pickle.dump(train_data, handle)

# Save the new csv file
'''
with open('train.csv', 'w') as csvfile:
    fieldnames = ['img_path', 'angle']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for key, value in train_data.items():
           writer.writerow([key, value])
'''

# Visulize the effect of smoothing
if img2show == True:

    plt.figure(1)
    plt.plot(range(Show_num), y[-Show_num:], 'ro')
 
    plt.figure(2)
    plt.plot(range(Show_num), sm2[-Show_num:], 'ro')

    plt.show()
