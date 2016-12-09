#!/usr/locals/python
#author:Shanglin Yang(kudoysl@gmail.com)

import argparse
import base64
import json

import pickle
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
count = 0

image_list = []
angle_list = []


@sio.on('telemetry')
def telemetry(sid, data):
    
    global count     # Count the total image in test 
    global image_list  # Store the saved test image path
    global angle_list # Store the saved angle
    output_ele = {}   # To save the output

    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))

    IMG_HEIGHT = 160 
    IMG_Width  = 320
    #image = image.resize((IMG_Width,IMG_HEIGHT))
    #image.save('./7_output/'+str(count)+ ".jpg", "JPEG")  # save the current image
    count += 1
    
    data = np.asarray(image)
    
    RGB = data/255.0  # Preprocessing the image 
    Y = np.zeros(shape=(1,IMG_HEIGHT,IMG_Width,3))
    Y [0,:,:,:] = RGB
    #Y[0,:,:,0] = 0.299 * RGB[:,:,0] + 0.587 * RGB[:,:,1] + 0.114 * RGB[:,:,2]  # Calculate the Y value
    
    
    transformed_image_array = Y
    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    angle,value = model.predict(transformed_image_array, batch_size=1)  # Model predict

    print ("angle: ",angle)
    
    direct = np.argmax(angle[0]) # The direct of the car (0 : turn left, 1: straight, 2: turn right)
    if direct == 1:
        steering_angle = 0
    elif direct == 0:
        min_value = min(0.3,value[0][0]) # This may be removed in the future if speed needed increases
        steering_angle = (-1)*min_value
    elif direct == 2:
        min_value = min(0.3,value[0][0])
        steering_angle = float(min_value)
        

    throttle = 0.2
    
    # Save the images
    image_list.append('./7_output/'+str(count)+ ".jpg")
    angle_list.append(float(steering_angle))
    output_ele['img_path'] = image_list
    output_ele['angle'] = angle_list

    #Save the pickle file for predictions
    #with open("output.p", "wb") as fi:
        #pickle.dump(output_ele, fi)
    #pickle.dump(output_ele, open("output.p","ab"))

    print(steering_angle, throttle,speed)
    send_control(float(speed)*steering_angle/15.0, throttle)  #adjust the result based on the speed 


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)