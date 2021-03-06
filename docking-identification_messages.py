#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D,MaxPooling2D,Flatten,Conv1D
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import json
import numpy as np
from sklearn import preprocessing
import keras
from keras.layers import Input,Dense,Lambda
from keras.models import Model
import os
import numpy as np
from keras.preprocessing import image as image_p
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pickle
import matplotlib.image as mpimg
from keras.preprocessing import image
from  matplotlib import pyplot
from keras.layers.normalization import BatchNormalization
import cv2
import random
from PIL import Image
from sklearn.utils import class_weight
from keras.layers import Reshape
import keras.backend as K
import math


#import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split


import json
import numpy as np
from sklearn import preprocessing
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.image as mpimg
from  matplotlib import pyplot
import random
from PIL import Image
from sklearn.utils import class_weight
import numpy

import math



import json
import numpy as np
from sklearn import preprocessing
import keras
from keras.layers import Input,Dense,Lambda,RepeatVector,Dot
from keras.models import Model
import os
import numpy as np
from keras.preprocessing import image as image_p
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pickle
import matplotlib.image as mpimg
from keras.preprocessing import image
from  matplotlib import pyplot
from keras.layers.normalization import BatchNormalization
import cv2
import random
from PIL import Image
from sklearn.utils import class_weight
from keras.layers import Reshape,merge,Concatenate,Add,Dropout
import keras.backend as K
import math
from keras.activations import softmax,tanh
import tensorflow as tf
from keras.applications.vgg16 import VGG16


from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D,MaxPooling2D,Flatten,Conv1D,Softmax
from keras.preprocessing import sequence
import socket
import time
import cv2
import numpy as np
import sys
import os
from datetime import datetime


from tensorflow.python.client import device_lib
device_lib.list_local_devices()



server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.settimeout(0.2)




def process_image_keypoints(img,bbox_coords):
    desired_size = 224

    old_size = img.shape

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    bbox_coordinates = []
    for a,b in bbox_coords:
        a = float(a)*ratio
        b = float(b)*ratio
        bbox_coordinates.append([a+left,b+top])
    return new_im,bbox_coordinates


# In[6]:


def process_image_keypoints_nobox(img):
    desired_size = 224

    old_size = img.shape

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)

    return new_im,[left,top,ratio]


# In[10]:


cnn_base = VGG16(input_shape=(224,224,3),include_top=False,weights='imagenet')

for layer in cnn_base.layers:
    #if layer.name=='block5_conv3':
        #break
    layer.trainable = False

x = cnn_base.get_layer('block5_pool').output
x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(64, kernel_size=(3, 3), padding="same")(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)
x = Conv2D(4, kernel_size=(1, 1), padding="same")(x)
x = Reshape((4,),name='localization_output')(x)


model = Model(inputs=cnn_base.input, outputs=x)
sgd = keras.optimizers.SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="mean_squared_error", optimizer=sgd, metrics=['accuracy'])
initial_weights = model.get_weights()


with open('vgg16-docker-box-weights-underwater-camera.hd5','rb') as f:
    model_weights = pickle.load(f)
    model.set_weights(model_weights)
    print("loaded weights")








class Message:
    def __init__(self):    
        self.message_type = "001"
        self.date = ""
        self.time = ""
        self.target_range = -10
        self.target_heading = 0
        self.elevation = 0
        self.target_quality = 1
        self.number_of_lights = -1
        self.camera_status = 0
        self.checksum = ""
    def set_number_of_lights(self,num_lights):
        self.number_of_lights = num_lights       
    def set_camera_status(self,success):
    	if success==True:
    		self.camera_status = 1
    	else:
    	 	self.camera_status = -1        
    def convert_to_string(self):
    	return ",".join(["$PISE","CAM2VCC",self.message_type,self.date,self.time,"{0:.1f}".format(self.target_range),"{0:.1f}".format(self.target_heading),"{0:.1f}".format(self.elevation),str(self.target_quality),str(self.number_of_lights),str(self.camera_status),self.checksum])	
    def set_date_time(self):
    	self.date = datetime.now().strftime('%Y%m%d')
    	self.time = datetime.now().strftime('%H:%M:%S.%f')[:-4]
    def fill_target_heading_elevation(self,translation_vector,rotation_vector):
    	x = np.array([translation_vector[0][0],translation_vector[1][0],translation_vector[2][0]])
    	self.target_range = np.linalg.norm(x)/100#convert cm to m
    	self.target_heading =  rotation_vector[0][0]
    	self.elevation =  rotation_vector[1][0]
    def set_checksum(self):
    	chk_sum_string = "".join(["$PISE","CAM2VCC",self.message_type,self.date,self.time,"{0:.1f}".format(self.target_range),"{0:.1f}".format(self.target_heading),"{0:.1f}".format(self.elevation),str(self.target_quality),str(self.number_of_lights),str(self.camera_status)])
    	calc_cksum = 0
    	for s in chk_sum_string:
    	    	calc_cksum ^= ord(s)
    	self.checksum = str(hex(calc_cksum)).lstrip("0").lstrip("x")

class Message2:
    def __init__(self):    
        self.message_type = "002"
        self.date = ""
        self.time = ""
        self.number_lights = 0
        self.light1_x_pos = 0
        self.light1_y_pos = 0
        self.light2_x_pos = 0
        self.light2_y_pos = 0
        self.light3_x_pos = 0
        self.light3_y_pos = 0
        self.light4_x_pos = 0
        self.light4_y_pos = 0
        self.light5_x_pos = 0
        self.light5_y_pos = 0
        self.light6_x_pos = 0
        self.light6_y_pos = 0
        self.camera_status = 0
        self.checksum = ""
    def set_number_of_lights(self,num_lights):
        self.number_of_lights = num_lights                  
    def set_camera_status(self,success):
    	if success==True:
    		self.camera_status = 1
    	else:
    	 	self.camera_status = -1
    def fill_light_positions(self,image_points):
    	self.number_lights = len(image_points)
    	if len(image_points)>=1:
    		self.light1_x_pos = image_points[0][0]
    		self.light1_y_pos = image_points[0][1]
    	if len(image_points)>=2:
    		self.light2_x_pos = image_points[1][0]
    		self.light2_y_pos = image_points[1][1]
    	if len(image_points)>=3:
    		self.light3_x_pos = image_points[2][0]
    		self.light3_y_pos = image_points[2][1]
    	if len(image_points)>=4:
    		self.light4_x_pos = image_points[3][0]
    		self.light4_y_pos = image_points[3][1]
    	if len(image_points)>=5:
    		self.light5_x_pos = image_points[4][0]
    		self.light5_y_pos = image_points[4][1]
    	if len(image_points)>=6:
    		self.light6_x_pos = image_points[5][0]
    		self.light6_y_pos = image_points[5][1]
    def convert_to_string(self):
        print(self.message_type)
        return ",".join(["$PISE","CAM2VCC",self.message_type,self.date,self.time,str(self.number_lights),"{0:.1f}".format(self.light1_x_pos),"{0:.1f}".format(self.light1_y_pos),"{0:.1f}".format(self.light2_x_pos),"{0:.1f}".format(self.light2_y_pos),"{0:.1f}".format(self.light3_x_pos),"{0:.1f}".format(self.light3_y_pos),"{0:.1f}".format(self.light4_x_pos),"{0:.1f}".format(self.light4_y_pos),"{0:.1f}".format(self.light5_x_pos),"{0:.1f}".format(self.light5_y_pos),"{0:.1f}".format(self.light6_x_pos),"{0:.1f}".format(self.light6_y_pos),str(self.camera_status),str(self.checksum)])
    def set_date_time(self):
    	self.date = datetime.now().strftime('%Y%m%d')
    	self.time = datetime.now().strftime('%H:%M:%S.%f')[:-4]	        
    def set_checksum(self):
    	chk_sum_string = "".join(["$PISE","CAM2VCC",self.message_type,self.date,self.time,str(self.number_lights),"{0:.1f}".format(self.light1_x_pos),"{0:.1f}".format(self.light1_y_pos),"{0:.1f}".format(self.light2_x_pos),"{0:.1f}".format(self.light2_y_pos),"{0:.1f}".format(self.light3_x_pos),"{0:.1f}".format(self.light3_y_pos),"{0:.1f}".format(self.light4_x_pos),"{0:.1f}".format(self.light4_y_pos),"{0:.1f}".format(self.light5_x_pos),"{0:.1f}".format(self.light5_y_pos),"{0:.1f}".format(self.light6_x_pos),"{0:.1f}".format(self.light6_y_pos),str(self.camera_status)])
    	calc_cksum = 0
    	for s in chk_sum_string:
    	    	calc_cksum ^= ord(s)
    	self.checksum = str(hex(calc_cksum)).lstrip("0").lstrip("x")



vidcap = cv2.VideoCapture('stream3.mp4')
success,image = vidcap.read()
count = 0
success = True
path = "stream2_images"
print(path)
try:
    os.mkdir(path);
except:
    print("exsits")



while success:
    m1 = Message()
    m2 = Message2()
    try:
        img_temp,corr = process_image_keypoints_nobox(image)
        img_temp = (img_temp[:,:,:] / 255.0).astype(np.float64)
        

        [[xmin,ymin,xmax,ymax]] = model.predict(np.expand_dims(img_temp,0))
        [left,top,ratio] = corr
        xmin,xmax = (xmin - left)/ratio,(xmax - left)/ratio
        ymin,ymax = (ymin - top)/ratio,(ymax - top)/ratio            

        img = image[int(ymin):int(ymax),int(xmin):int(xmax),:]
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret,th1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)

        pt1 = (int(xmin),int(ymin))
        pt2 = (int(xmax),int(ymax))
        image = cv2.rectangle(image,pt1,pt2,(0,255,255),thickness=2)

        tuples = []
        contour_count = 0
        exception_count = 0
        ret,contours,hierachy = cv2.findContours(th1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        for c in contours:
            try:
                M = cv2.moments(c)

                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                tuples.append((cX+int(xmin),cY+int(ymin)))
                contour_count = contour_count + 1
            except:
                exception_count = exception_count + 1        
        final_tuples = []
        contour5_flag = False
        if contour_count>5:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=0).fit(tuples)
            for tup in kmeans.cluster_centers_:
                image = cv2.circle(image,(int(tup[0]),int(tup[1])), 5, (255,0,0), -1)
                final_tuples.append((int(tup[0]),int(tup[1])))
            logger.info("more than 5 landmarks clustering")
            contour5_flag = True
        else:
            if len(tuples)==5:
                logger.info("equal to five landmarks found")
                contour5_flag = True
            else:
                logger.info("less than five landmarks found")
                contour5_flag = False
            for tup in tuples:
                image = cv2.circle(image,(tup[0],tup[1]), 5, (255,0,0), -1)
                final_tuples.append(tup)
                
            
        
        image_points = np.array(sorted(final_tuples, key=lambda tup: tup[0]),dtype="double")    
        print("Sorted points",image_points)
        m2.fill_light_positions(image_points)
        m1.set_number_of_lights(len(image_points))
        m2.set_number_of_lights(len(image_points))
        if contour5_flag==True:
            model_points = np.array([
                                        (-40, 13, 0.0),
                                        (-25,-34, 0.0),
                                        (0, 42, 0.0), 
                                        (25,-34, 0.0),
                                        (40,13, 0.0)
                                    ])
            # Camera internals

            size = image.shape
            focal_length = size[1]
            center = (size[1]/2, size[0]/2)
            camera_matrix = np.array(
                                     [[focal_length, 0, center[0]],
                                     [0, focal_length, center[1]],
                                     [0, 0, 1]], dtype = "double"
                                     )

            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)



            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose


            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 42.0, 50.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            print(nose_end_point2D)
            for p in image_points:
            	cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

            print(rotation_vector)
            print(translation_vector)

            text= "position in cm:"+str(round(translation_vector[0][0],1))+","+str(round(translation_vector[1][0],1))+","+str(round(translation_vector[2][0],1))
            cv2.putText(image,text,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),2,cv2.LINE_AA)
            
            text= "yaw,pitch,roll in deg:"+ str(round(rotation_vector[0][0],1))+","+str(round(rotation_vector[1][0],1))+","+str(round(rotation_vector[2][0],1))
            cv2.putText(image,text,(20,image.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),2,cv2.LINE_AA)

            
            p1 = ( int(image_points[2][0]), int(image_points[2][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(image, p1, p2, (255,0,0), 2)
          

    	    
            	
            #cv2.imwrite(string, image) 	   
    
        
        #break
        m1.fill_target_heading_elevation(translation_vector,rotation_vector)
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
       		break
    except Exception as ex:
        print(ex)
    m1.set_date_time()
    m1.set_checksum()
    m1.set_camera_status(success)
    m2.set_date_time()
    m2.set_checksum()
    m2.set_camera_status(success)
    logger.info("Message 1 is %s",m1.convert_to_string())
    logger.info("Message 2 is %s",m2.convert_to_string())
    
    
    server.sendto(m1.convert_to_string().encode(), ('<broadcast>', 51110))
    server.sendto(m2.convert_to_string().encode(), ('<broadcast>', 51120))
    
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




