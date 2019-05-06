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
import pickle
from utils import WeightReader, decode_netout, draw_boxes, Message,Message2
from numpy.linalg import inv

import json
import numpy as np
from sklearn import preprocessing
from keras.layers import Input,Dense,Lambda
from keras.models import Model
from keras.preprocessing import image as image_p
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
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
from tensorflow.python.client import device_lib
from keras.applications.vgg16 import VGG16


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
from keras.models import model_from_json

from sklearn.cluster import KMeans
import argparse
import sys
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


#device_lib.list_local_devices()

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.degrees(np.array([x, y, z]))


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

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
            

def send_camera_error_messages(server,port1,port2):
	m1 = Message()
	m2 = Message()

	m1.set_date_time()
	m1.set_camera_status(-1)
	m1.set_checksum()

	m2.set_date_time()
	m2.set_camera_status(-1)
	m2.set_checksum()
	logger.info("Message 1 is %s",m1.convert_to_string())
	logger.info("Message 2 is %s",m2.convert_to_string())

	print("Sending m1 to port:"+port1)
	server.sendto(m1.convert_to_string().encode(), ('<broadcast>', int(port1)))
	print("Sending m2 to port:"+port2)
	server.sendto(m2.convert_to_string().encode(), ('<broadcast>', int(port2)))
print("----------------------------------------------------------------------------------------------------")









ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weight", type=str, required=True,help="weights")
ap.add_argument("-cfg", "--config", type=str, required=True,help="configuration")
ap.add_argument("-s", "--size", type=str, required=True,help="big or small")
ap.add_argument("-src", "--source", type=str, required=False,choices=['video','stream'],help="video or stream")
ap.add_argument("-l", "--lights", type=str, required=False,choices=['True','False'],help="show light coordinates or not")
ap.add_argument("-save_train", "--save_train", type=str, required=False,choices=['True','False'],help="save data to folder for training")
ap.add_argument("-save_video", "--save_video", type=str, required=False,choices=['True','False'],help="save data to folder for creating video")
ap.add_argument("-skip", "--skip", type=str, required=False,help="process every nth frame")
ap.add_argument("-video", "--video", type=str, required=False,help="video for processing")
ap.add_argument("-slink", "--stream_link", type=str, required=False,help="stream link")
ap.add_argument("-m", "--medium", type=str, required=False,choices=['air','water'],help="choose the camera medium")
ap.add_argument("-p1", "--port1", type=str, required=False,help="port 1")
ap.add_argument("-p2", "--port2", type=str, required=False,help="port 2")
ap.add_argument("-out", "--output_file", type=str, required=False,help="redirect to output file")
ap.add_argument("-conn_test", "--conn_test", type=str, required=False,choices=['True','False'],help="Test connection before startup")
args = vars(ap.parse_args())
print(args)

weights_path = args["weight"]
config_path = args["config"]
hoop_size = args["size"]
source =  "stream" if args["source"]==None else args["source"]
show_lights = True if args["lights"]=='True' else False
multiplier = 5 if args["skip"]==None else int(args["skip"])
save_data_for_training = True if args["save_train"]=='True' else False
save_data_for_video = True if args["save_video"]=='True' else False     
video_link =  "stream2.mp4" if args["video"]==None else args["video"]
default_stream_link = 'rtsp://admin:admin@192.168.214.40/h264.sdp?res=half&x0=0&y0=0&x1=1920&y1=1080&qp=16&doublescan=0&ssn=41645' 
stream_link = default_stream_link if args["stream_link"]==None else args["stream_link"]
medium =  "air" if args["medium"]==None else args["medium"]
port1 =  "51110" if args["port1"]==None else args["port1"]
port2 =  "51120" if args["port2"]==None else args["port2"]
conn_test =  True if args["conn_test"]=='True' else False
output_file = "output.txt" if args["output_file"]==None else args["output_file"]


print(weights_path)
print(config_path)
print(hoop_size)
print("Source is",source)
print("show lights is",show_lights)
print("Processing every nth frame",multiplier)
print(args)



#sys.stdout = open(output_file, 'w')
sys.stdout = open(os.devnull, 'w')
print('printing to file',output_file)

server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.settimeout(0.2)


LABELS = ['station']
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
#ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS          = [0.1,0.1,0.7,0.7,2,2,5,5,9,9]
TRUE_BOX_BUFFER  = 50




model_points_big = np.array([
                                        (-40, 13, 0.0),
                                        (-25,-34, 0.0),
                                        (0, 42, 0.0), 
                                        (25,-34, 0.0),
                                        (40,13, 0.0)
                                    ])

model_points_small = np.array([
                                        (-32, 11, 0.0),
                                        (-20,-28, 0.0),
                                        (0, 34, 0.0), 
                                        (20,-28, 0.0),
                                        (32,11, 0.0)
                                    ])
if source=="stream":
	if conn_test==True:
		logger.info("Testing camera connection")
		while(1):
			vidcap = cv2.VideoCapture(stream_link)
			print(vidcap)
			if vidcap is None or not vidcap.isOpened():
				print(vidcap)
				logger.info('Warning: unable to open video source: ',stream_link)
				send_camera_error_messages(server,port1,port2)
			else:
				logger.info("Camera is connected")
				break
	else:
		vidcap = cv2.VideoCapture(stream_link)
else:
	vidcap = cv2.VideoCapture(video_link)
	        
      
if hoop_size=="small":
    model_points = model_points_small
elif hoop_size=="big":
    model_points = model_points_big
elif RepresentsInt(hoop_size)==True:
    radius = int(hoop_size)/2
    startx = 0
    starty = 0

    start_ang = -90

    model_points = []
    for i in range(5):
        ang = start_ang + (i*72)
        startx = radius*np.cos(np.deg2rad(ang))
        starty = -radius*np.sin(np.deg2rad(ang))
        #print(np.round(startx,0),np.round(starty,0))
        model_points.append((np.round(startx,0),np.round(starty,0),0.0))
    #print(final_tuples)
    model_points = np.array(sorted(model_points, key=lambda tup: tup[0]),dtype="double")
    #print(image_points)        
else:
    logger.info("Invalid diameter")
    raise ValueError('Invalid diameter')
    
        
         
    
print("Model config",model_points)    

# In[10]:

with open(config_path,'rb') as f:
    cfg = pickle.load(f)
    model = model_from_json(cfg)
with open(weights_path,'rb') as f:
	weights = pickle.load(f)
	model.set_weights(weights)
avail = True
print("GPU available")
   
seconds = 0.5
#fps = vidcap.get(cv2.CAP_PROP_FPS) # Gets the frames per second



success,image = vidcap.read()
#image = cv2.imread('/home/nvidia/docking/imgg.jpg')
count = 0
success = True
path = "stream2_images"
print(path)
try:
    os.mkdir(path);
except:
    print("exsits")


dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))

while success:
    m1 = Message()
    m2 = Message2()
    try:
        frameId = int(round(vidcap.get(1)))
        if not frameId % multiplier==0:
                raise ValueError('Skipping frame id',str(frameId))
        #img_temp,corr = process_image_keypoints_nobox(image)
        #img_temp = (img_temp[:,:,:] / 255.0).astype(np.float64)
        
        
        
        
        input_image = cv2.resize(image, (224, 224))
        input_image = input_image / 255.
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)

        netout = model.predict([input_image, dummy_array])
        boxes = decode_netout(netout[0],obj_threshold=0.1,nms_threshold=0.2,anchors=ANCHORS,nb_class=CLASS)            
        
        max_score = -1
        saved_box = None
        for bbox in boxes:
                if bbox.get_score()>max_score:
                        saved_box = bbox
        image_h, image_w, _ = image.shape       
        if saved_box == None:
                m1.set_target_detected(0)
                img = image
                logger.info('Target not detected')        
        else:
                xmin = int(saved_box.xmin*image_w)
                ymin = int(saved_box.ymin*image_h)
                xmax = int(saved_box.xmax*image_w)
                ymax = int(saved_box.ymax*image_h)            
                
                pt1 = (int(xmin),int(ymin))
                pt2 = (int(xmax),int(ymax))
                
                img = image[int(ymin):int(ymax),int(xmin):int(xmax),:]
        
        

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret,th1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
        #th1 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,105,2)
        ret, th1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #cv2.imwrite("imgg.jpg",img)
        #break
        

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
                if saved_box!=None:
                        tuples.append((cX+int(xmin),cY+int(ymin)))
                else:
                        tuples.append((cX,cY))
                contour_count = contour_count + 1
            except:
                exception_count = exception_count + 1        
        final_tuples = []
        contour5_flag = False
        if contour_count>5:
            kmeans = KMeans(n_clusters=5, random_state=0).fit(tuples)
            for tup in kmeans.cluster_centers_:
                image = cv2.circle(image,(int(tup[0]),int(tup[1])), 5, (255,0,0), -1)
                final_tuples.append((int(tup[0]),int(tup[1])))
            print("more than 5 landmarks clustering")
            contour5_flag = True
        else:
            if len(tuples)==5:
                print("equal to five landmarks found")
                contour5_flag = True
            else:
                print("less than five landmarks found")
                contour5_flag = False
            for tup in tuples:
                image = cv2.circle(image,(tup[0],tup[1]), 5, (255,0,0), -1)
                final_tuples.append(tup)
                
            
        
        image_points = np.array(sorted(final_tuples, key=lambda tup: tup[0]),dtype="double")    
        print("Sorted points",image_points)
        
        if saved_box != None:
                        image = cv2.rectangle(image,pt1,pt2,(0,255,255),thickness=2)
            
        image_points_altered = [[],[],[],[],[]]
        
        if contour5_flag==True:
                if image_points[0][1]>image_points[1][1]:
                        image_points_altered[0] = (image_points[1][0],image_points[1][1])
                        image_points_altered[1] = (image_points[0][0],image_points[0][1])
                else:
                        image_points_altered[0] = (image_points[0][0],image_points[0][1])
                        image_points_altered[1] = (image_points[1][0],image_points[1][1])                         
                if image_points[2][1]<image_points[3][1] and image_points[2][1]<image_points[4][1]:
                        image_points_altered[2] = (image_points[2][0],image_points[2][1])
                        if image_points[3][1] < image_points[4][1]:
                                print("Config 1")
                                image_points_altered[3] = (image_points[4][0],image_points[4][1])
                                image_points_altered[4] = (image_points[3][0],image_points[3][1])
                        else:
                                print("Config 2")
                                image_points_altered[3] = (image_points[3][0],image_points[3][1])
                                image_points_altered[4] = (image_points[4][0],image_points[4][1])
                elif image_points[2][1]>image_points[3][1] and image_points[2][1]>image_points[4][1]:
                        image_points_altered[3] = (image_points[2][0],image_points[2][1])
                        if image_points[3][1] < image_points[4][1]:
                                print("Config 3")
                                image_points_altered[2] = (image_points[3][0],image_points[3][1])
                                image_points_altered[4] = (image_points[4][0],image_points[4][1])
                        else:
                                print("Config 4")
                                image_points_altered[2] = (image_points[4][0],image_points[4][1])
                                image_points_altered[4] = (image_points[3][0],image_points[3][1])
                else:
                        logger.error("Invalid pentagon shape")
                        image_points_altered[2] = (image_points[2][0],image_points[2][1])
                        image_points_altered[3] = (image_points[3][0],image_points[3][1])
                        image_points_altered[4] = (image_points[4][0],image_points[4][1])                
        
        image_points = np.array(image_points_altered)        
        print("Sorted altered points",image_points)      
        m1.set_number_of_lights(len(image_points))
        m2.set_number_of_lights(len(image_points))
        
        if contour5_flag==True:
            print("entered")
            
            # Camera internals

            size = image.shape
            #focal_length = size[1]
            #center = (size[1]/2, size[0]/2)
            focal_length = 780
            center = (479.72, 104.56)
            camera_matrix = np.array(
                                     [[768.31, 0, center[0]],
                                     [0, 768.21, center[1]],
                                     [0, 0, 1]], dtype = "double"
                                     )

            #dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            #dist_coeffs = np.zeros((5,1)) # Assuming no lens distortion
            dist_coeffs = np.array([[-0.3023023,0.14315257,-0.00201115,-0.00041268,-0.04129913]])
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if medium=="water":
                        print("Scaling translation vector by 1.33") 
                        translation_vector = translation_vector*1.33        
            print("euler calcluting")
            print(rotation_vector)
            rot_mat, _ = cv2.Rodrigues(rotation_vector)
            print(rot_mat)  
            euler_angles = rotationMatrixToEulerAngles(rot_mat)
            print("euler calculated")



            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose


            light_camera_coordinates = []
            print("Rotation matrix",rot_mat)
            print("Translation vector",translation_vector)

            transformation_matrix = np.eye(4)
            transformation_matrix[:3,:3] = np.array(rot_mat)
            print("Transformation matrix",transformation_matrix) 
            transformation_matrix[:3,3] = [translation_vector[0][0],translation_vector[1][0],translation_vector[2][0]]
            print("Transformation matrix",transformation_matrix)
            
            #try:
            #            inv_rot_mat  = inv(rot_mat);
            #            inv_camera_matrix  = inv(cameraMatrix) 
            #            rightSideMat = inv_rot_mat* tvec;
            for idx in range(model_points.shape[0]):
                        light_coords = np.matmul(transformation_matrix,np.append(model_points[idx],1).reshape(4,1))
                        print(light_coords)
                        light_camera_coordinates.append(light_coords[:3]) 
            for idx,p in enumerate(image_points):
            	cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            	if show_lights==True:
                    	text = str(round(light_camera_coordinates[idx][0][0],1))+","+str(round(light_camera_coordinates[idx][1][0],1))
                    	cv2.putText(image,text,(int(p[0])-1, int(p[1])-1), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),2,cv2.LINE_AA)

            
            print("Euler angles ",euler_angles)
            print("Translation angles",translation_vector)
            heading = np.degrees(np.arctan2(translation_vector[0][0],translation_vector[2][0]))
            elevation = np.degrees(np.arctan2(translation_vector[1][0],translation_vector[2][0]))
            m1.fill_target_heading_elevation(translation_vector,heading,elevation)
            m2.fill_light_positions(light_camera_coordinates)
            
        

           
            text= "position in cm:"+str(round(translation_vector[0][0],1))+","+str(round(translation_vector[1][0],1))+","+str(round(translation_vector[2][0],1))
            cv2.putText(image,text,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),2,cv2.LINE_AA)
            
            
            text= "distance,heading,elevation in m,deg,deg:"+str(round(np.linalg.norm(translation_vector)/100,2))+","+str(round(heading,1))+","+str(round(elevation,1))
            
            cv2.putText(image,text,(20,image.shape[0]-70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),2,cv2.LINE_AA)
            text= "yaw,pitch,roll in deg:"+ str(round(euler_angles[0],1))+","+str(round(euler_angles[1],1))+","+str(round(euler_angles[2],1))
            cv2.putText(image,text,(20,image.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,0,0),2,cv2.LINE_AA)

            
            #p1 = ( int(image_points[2][0]), int(image_points[2][1]))
            #p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            #cv2.line(image, p1, p2, (255,0,0), 2)
          

            #cv2.imwrite("string.jpg", image)
            #plt.imshow(image) 	   
            #break
        
        #break
    except Exception as ex:
        print(ex)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    m1.set_date_time()
    m1.set_camera_status(1)
    m1.set_checksum()
    m2.set_date_time()
    m2.set_camera_status(1)
    m2.set_checksum()
    
    
    
    if frameId % multiplier==0:
        cv2.imshow('frame',image)
        count = count + 1 
        logger.info("Message 1 is %s",m1.convert_to_string())
        logger.info("Message 2 is %s",m2.convert_to_string())
    
        print("Sending m1 to port:"+port1)
        server.sendto(m1.convert_to_string().encode(), ('<broadcast>', int(port1)))
        print("Sending m2 to port:"+port2)
        server.sendto(m2.convert_to_string().encode(), ('<broadcast>', int(port2)))
    
    
    if save_data_for_video==True and frameId % multiplier==0:
        cv2.imwrite("stream_video/"+str(int(time.time()))+".jpg",image)
        
    success,image = vidcap.read()
    
    
    if save_data_for_training==True and frameId % multiplier==0:
        cv2.imwrite("stream_training/"+str(int(time.time()))+".jpg",image)
        
    print('Read a new frame: ', success)
    
sys.exit()

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




