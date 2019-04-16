import cv2
import numpy as np
import glob
import os
 
img_array = []
dirFiles= os.listdir("stream4_images/")
#dirFiles.sort(key=lambda f: int(filter(str.isdigit, f)))
dirFiles = sorted(dirFiles,key=lambda x: int(os.path.splitext(x)[0]))
for filename in dirFiles:
    print(filename)
    img = cv2.imread("stream4_images/"+filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('new_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
