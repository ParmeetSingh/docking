import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2
from datetime import datetime

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin*image_w)
        ymin = int(box.ymin*image_h)
        xmax = int(box.xmax*image_w)
        ymax = int(box.ymax*image_h)

        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
        cv2.putText(image, 
                    labels[box.get_label()] + ' ' + str(box.get_score()), 
                    (xmin, ymin - 13), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image_h, 
                    (0,255,0), 2)
        
    return image          
        
def decode_netout(netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold
    
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    
                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes    

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua  
    
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap      
        
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)
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
    def set_target_detected(self,target_quality):
        self.target_quality = target_quality    
    def set_number_of_lights(self,num_lights):
        self.number_of_lights = num_lights       
    def set_camera_status(self,status):	
    	self.camera_status = status        
    def convert_to_string(self):
    	return ",".join(["$PISE","CAM2VCC",self.message_type,self.date,self.time,"{0:.1f}".format(self.target_range),"{0:.1f}".format(self.target_heading),"{0:.1f}".format(self.elevation),str(self.target_quality),str(self.number_of_lights),str(self.camera_status),self.checksum])	
    def set_date_time(self):
    	self.date = datetime.now().strftime('%Y%m%d')
    	self.time = datetime.now().strftime('%H:%M:%S.%f')[:-4]
    def fill_target_heading_elevation(self,translation_vector,heading,elevation):
    	x = np.array([translation_vector[0][0],translation_vector[1][0],translation_vector[2][0]])
    	self.target_range = np.linalg.norm(x)/100#convert cm to m
    	self.target_heading =  heading
    	self.elevation =  elevation
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
    def set_camera_status(self,status):
        self.camera_status = status
    def fill_light_positions(self,image_points):
    	self.number_lights = len(image_points)
    	if len(image_points)>=1:
    		self.light1_x_pos = image_points[0][0][0]
    		self.light1_y_pos = image_points[0][1][0]
    	if len(image_points)>=2:
    		self.light2_x_pos = image_points[1][0][0]
    		self.light2_y_pos = image_points[1][1][0]
    	if len(image_points)>=3:
    		self.light3_x_pos = image_points[2][0][0]
    		self.light3_y_pos = image_points[2][1][0]
    	if len(image_points)>=4:
    		self.light4_x_pos = image_points[3][0][0]
    		self.light4_y_pos = image_points[3][1][0]
    	if len(image_points)>=5:
    		self.light5_x_pos = image_points[4][0][0]
    		self.light5_y_pos = image_points[4][1][0]
    	if len(image_points)>=6:
    		self.light6_x_pos = image_points[5][0][0]
    		self.light6_y_pos = image_points[5][1][0] 
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
def ping_to_stream(hostname):
        response = os.system("ping -c 1 " + hostname)
        
        return response    	      
