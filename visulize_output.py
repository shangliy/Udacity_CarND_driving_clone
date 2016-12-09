#!/usr/locals/python
#author:Shanglin Yang(kudoysl@gmail.com)

'''
This script is to visulize the center image and corresponding angle value
The data being smoothed using 'pyasl.smooth'
'''

import PIL
import json
import pickle
import heapq
import tkinter
from tkinter import *
from PIL import Image
from PIL import ImageTk 
from tkinter import filedialog

import sys
import numpy as np
from scipy.misc import imread, imresize


IMG_WEIGHT = 299
IMG_HEIGHT = 299

class triplet_sequential():
    def __init__(self, Image_path,master,trinum):
        self.image_list, self.angle, self.num = self.data_load(Image_path)
        self.M = self.init_matrix(self.num)
        self.panelA = None
        self.textA = None
        self.root = master
        self.tri_num = trinum
        self.data_generator = self.visulize_img()

        self.anchor_path = []
        self.angle_arr = []

        self.currentangle = 0
        
        self.count = 0
        
    def init_matrix(self,img_num):   
        return np.zeros((img_num,img_num))

    def data_load(self,file_name):
        '''
        image_set = []
        angle_set = []
        with open(file_name,"rb") as J:
            output_data = pickle.load(J)
            image_set = (output_data["img_path"])
            angle_set = (output_data["angle"])  
              
        
        num = len(image_set)
        return image_set, angle_set, num
        '''

        image_set = []
        angle_set = []
        with open(file_name,"rb") as J:
            output_data = pickle.load(J)
            image_set = (output_data["img_path"])
            angle_set = (output_data["angle"])  
              
        
        num = len(image_set)
        return image_set, angle_set, num

    def increase_angle(self):
        
        self.currentangle += 0.02
        self.textA.configure(text=str(self.currentangle))

    def decrease_angle(self):
        
        self.currentangle -= 0.02
        self.textA.configure(text=str(self.currentangle))

    def next_image(self):
        self.count += 1
        self.show_next()

    def save_json(self):
        Data_Json = []
        num = 0
        print ("Total num is ", self.count)
        while(num < self.count):
            json_dict = {}
            # Getting a pair of clients
            json_dict["path"] = str(self.anchor_path[num])
            json_dict["angle"] = float(self.angle_arr[num])
            Data_Json.append(json_dict)
            num = num + 1
        
        outfile = open('new_ouput.json', 'w')
        json.dump(Data_Json, outfile,sort_keys = True, indent = 4)
        outfile.close()
        sys.exit()
        
    def show_next(self):
        
        self.angle_arr.append(float(self.currentangle))
        image_a, self.currentangle = next(self.data_generator)
        
        self.panelA.configure(image=image_a)
        self.textA.configure(text=str(self.currentangle))
        print('next image')

    def update_similar_matrix(self,i,j):
        
        return M
        

    def search_similar_img(self,M,anc_index):
        similar_array = M[anc_index,:]
       
        top_index = heapq.nlargest(1000, range(len(similar_array[0])), similar_array.take)
        np.random.shuffle(top_index)
        
        pos = top_index[1]
        
        neg = top_index[10]
        return pos, neg
        

    def visulize_img(self):

        while(1):
            for i in range(2000,self.num):   
                img_index = i
                #self.image_list[img_index] = './new_output/'+str(count)+ ".jpg"
                print(self.image_list[img_index])
                anchor_img = Image.open(self.image_list[img_index])
                self.anchor_path.append(self.image_list[img_index])
                
                image_a = ImageTk.PhotoImage(anchor_img)
                steer_angle = self.angle[i]
                yield image_a,steer_angle

    def show_image(self):
        
        # if the panels are None, initialize them
        if self.panelA is None:
            
            image_a,angle_a = next(self.data_generator)
            self.currentangle = angle_a
            
            # the first panel will store our original image
            self.panelA = Label(image=image_a,text='anchor')
            self.panelA.image = image_a
            self.panelA.pack( padx=10, pady=10)
            self.panelA.grid(row = 1, column = 0)

            self.textA = Label(text=str(angle_a))
            self.textA.config(font=( 'bold' ))
            self.textA.grid(row = 0, column = 0)

            btnadd = Button(root, text="increase the angle", command=self.increase_angle)
            btnadd.pack(fill="both", expand="yes", padx="10", pady="10")
            btnadd.grid(row = 2, column = 1)

            btndec = Button(root, text="decrease the angle", command=self.decrease_angle)
            btndec.pack(fill="both", expand="yes", padx="10", pady="10")
            btndec.grid(row = 2, column = 3)

            btnext = Button(root, text="next image", command=self.next_image)
            btnext.pack(fill="both", expand="yes", padx="10", pady="10")
            btnext.grid(row = 2, column = 2)

            btnsav = Button(root, text="save the final output", command=self.save_json)
            btnsav.pack(fill="both", expand="yes", padx="10", pady="10")
            btnsav.grid(row = 2, column = 0)


        # otherwise, update the image panels
        else:
            # update the pannels
            self.panelA.configure(image=image_a)
            self.panelA.image = image_a
        
if __name__ == "__main__":
    
    root = Tk()
    task = triplet_sequential("./output.p",root,1)

    app = task.show_image()

    root.mainloop()


