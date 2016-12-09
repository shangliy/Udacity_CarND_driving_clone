#!/usr/locals/python
#author:Shanglin Yang(kudoysl@gmail.com)

'''
This is the main script for training model.
Function:
    main(): Main function to do the training;
    model_build(): Build model structure;
    generate_arrays_from_file(): The batch data GeneratorExit;
    data_load(): Load the input file

'''


import csv
import sys
import os.path
import json
import pickle
import numpy as np


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from scipy.misc import imread, imresize
from keras.models import Sequential 
from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Reshape, BatchNormalization,Activation,Dropout
from keras.optimizers import Adam,RMSprop
from keras.utils.np_utils import to_categorical


# The input image size
IMG_HEIGHT = 160
IMG_WEIGHT = 320

#Batch size
batch_size = 32

def data_load(file_name):
    '''
    This function preload the image path and angle data
    Input: 
        file_name: path of training data list (pickle file),  containing the train image path and corresponding angle
    Output: 
        img_set: List of training images path
        val: Target steering angle
        count: Number of training data
    '''
    # Open pickle file 
    with open(file_name, "rb") as f:
        train_data = pickle.load(f)

    # Get the image path and target value
    image_path = train_data['img_path']
    y = train_data['angle']

    #Total train image data number
    num = len(image_path)

    count = 0
    img_set = []
    val = []

    # Using zero count to decrease the zero-value training data
    zero_count = 0

    for i in range(num):
        if abs(y[i]) < 0.04:  #Low pass filter to remove the noisy data
            zero_count += 1 
            if zero_count < 2: # choose one from two zero-value data
                continue
            else:
                zero_count = 0  
                val.append(float(0.0))  # Append the val data
                img_set.append(image_path[i])
                count += 1
        else:
            val.append(float((y[i])))
            img_set.append(image_path[i])
            count += 1

    return img_set, val , count


def generate_arrays_from_file(file_name,symbol):
    '''
    Traning/Validation data Generator
    Input: 
        file_name: path of training data list (pickle file),  containing the train image path and corresponding angle
        symbol: 'train': training data , 'validation': Validation data
    Output:
        A batch of training/validation data (x,[direct,value])
            x : preprocessed image data,shape:[batch_size,IMG_HEIGHT,IMG_WEIGHT,3]
            direct: Training direction, shape:[batch_size,3]
                0: turn left
                1: straight
                2: turn right
            value: Absolute value of steering angle, shape:[batch_size,1] , v:(0 to 1) 
    '''
    # Preload the required data index (path_list, val_list)
    image_path, y_t, data_num = data_load(file_name) 
    print("total image_num:",data_num)

    # Choose the training data lower and upped bound
    start = 0
    batch_num = (data_num - start)//batch_size #total batch num

    while 1:
        if (symbol == 'train'):
            sample_index = np.arange(start,data_num)
            np.random.shuffle(sample_index)
        else:
            sample_index = np.arange(0,data_num)
            np.random.shuffle(sample_index)
        
        x = np.zeros(shape=(batch_size,IMG_HEIGHT,IMG_WEIGHT,3),dtype='float32')
        value = np.zeros(shape=(batch_size,1),dtype='float32')
        direct = np.zeros(shape=(batch_size,3),dtype='int32')

        for j in range(batch_num):
            
            start_index = batch_size*j #Start offset

            for i in range(batch_size):
                
                # Get the image path and arget value
                center_img_path = image_path[sample_index[start_index+i]]
                y = float(y_t[sample_index[start_index+i]])
            
                # Read image data
                RGB = imresize(imread(center_img_path),(IMG_HEIGHT,IMG_WEIGHT))

                # Normalization of input data
                RGB = RGB/255.0 
                x[i,:,:,:] = RGB

                # Generate the direction and absolute value for the training

                if abs(y)<0.04: #Low pass filter to remove the noisy data
                    direct[i] = to_categorical([1], nb_classes=3) #Go straight, class = 1
                    value[i,0] = float(0.0)              
                else:
                    if (y>0):
                        direct[i] = to_categorical([2], nb_classes=3) # Turn right, class = 2 
                        value[i,0] = float(abs(y))
                    else: 
                        direct[i] = to_categorical([0], nb_classes=3) # Turn left, class = 0
                        value[i,0] = float(abs(y)) 

            yield (x,[direct,value])
 

def model_build():
    '''
    Build the deep learning network structure
    Return: 
        final_model: Keras Training model

    Structure:
     -------------------------------------------------------------------------------
     - Convolution2D layer, window_size(5x5), stride(2x2), depth: 32
     - Activation layer, 'relu'
     - BatchNormalization layer

     - Convolution2D layer, window_size(5x5), stride(2x2), depth: 64
     - Activation layer,'relu'
     - BatchNormalization layer

     - Convolution2D layer, window_size(5x5), stride(2x2), depth: 128
     - Activation layer,'relu'
     - BatchNormalization layer

     - Convolution2D layer, window_size(3x3), stride(1x1), depth: 128
     - Activation layer,'relu'
     - BatchNormalization layer

     - Convolution2D layer, window_size(3x3), stride(1x1), depth: 128
     - Activation layer,'relu'
     - BatchNormalization layer

     - Flaten layer
     - Dropout layer

     - Full connected layer : -> 100
     - Activation layer,'relu'
     - BatchNormalization layer

     - Full connected layer : 100 -> 50
     - Activation layer,'relu'
     - BatchNormalization layer

        - Full connected layer : 50 -> 3
        - Activation layer, 'softmax''

        - Full connected layer : 50 -> 10
        - Activation layer,'relu''
        Full connected layer : 10 -> 1

    ------------------------------------------------------------------------------------------
    Learning rate : Using Adam
    Loss function:
        Directiong loss: categorical_crossentropy
        Steering absolute value loss : 'mae'
    Loss weight : [1,1]

    '''
    # The key idea of the work is to make the in
    model = Sequential()
    # input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
    # this applies 32 convolution filters of size 3x3 each. 
    model.add(Convolution2D(32, 5, 5,subsample=(2, 2),border_mode='valid',input_shape=(IMG_HEIGHT, IMG_WEIGHT,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, 5, 5,subsample=(2, 2),border_mode='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(128, 5, 5,subsample=(2, 2),border_mode='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))


    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    fea = model.output

    logit = Dense(3)(fea)
    sign = Activation('softmax')(logit)

    fine_fea = Dense(10)(fea)
    new_fea = Activation('relu')(fine_fea)
    value_logit  = Dense(1)(fine_fea)

    final_model = Model(input = model.input, output=[sign,value_logit])

    final_model.compile(optimizer='adam', loss=['categorical_crossentropy','mae'], loss_weights=[1,1])

    return final_model

def main():
  
    # Set the training data
    train_data_path = "./train_a.p"  #Traning data path
    validation_path = "./test_a.p" #Traning validation path
    samples_pe  = 10000  #samples_per_epoch
    epoches = 10
    evaluation_num = 1000

    # Build the network
    final_model = model_build()

    # Load pre-trained model
    if (os.path.exists('./model.h5')):
        print("Load pre-trained model weights")
        final_model.load_weights('model.h5',by_name=True)

    # Training and evaluation generator
    train_generator = generate_arrays_from_file(train_data_path,'train')
    validation_generator = generate_arrays_from_file(validation_path,'validation')

    # train the model on the new data for a few epochs
    final_model.fit_generator(
            train_generator,
            samples_per_epoch=samples_pe//batch_size*batch_size,
            nb_epoch=epoches,
            validation_data=validation_generator,
            nb_val_samples=evaluation_num )

    #Save the model structure and weights
    json_string = final_model.to_json()
    with open('model.json', 'w') as outfile:
            json.dump(json_string, outfile)
    final_model.save_weights('model.h5')

    #Save the graph structure image
    from keras.utils.visualize_util import plot
    plot(final_model, to_file='model.png')

if __name__ == "__main__":
    main()

