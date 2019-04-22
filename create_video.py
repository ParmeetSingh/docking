import cv2
import numpy as np
import glob
import os
 
img_array = []
dirFiles= os.listdir("/home/nvidia/Videos/data/new/stream_training/")
#dirFiles.sort(key=lambda f: int(filter(str.isdigit, f)))
dirFiles = sorted(dirFiles,key=lambda x: int(os.path.splitext(x)[0]))
for filename in dirFiles:
    print(filename)
    img = cv2.imread("/home/nvidia/Videos/data/new/stream_training/"+filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
print("hewyy") 
out = cv2.VideoWriter('new_video2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
print("heww") 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
