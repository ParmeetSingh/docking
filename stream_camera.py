import numpy as np
import cv2



while(1):
	cap = cv2.VideoCapture("http://192.168.2.54:8080/video")
	print(cap)
	if cap is None or not cap.isOpened():
		print(cap)
		print('Warning: unable to open video source: ')
	else:
		print("Camera is connected")
		break	


count = 0
while(True):
    count = count + 1
    ret, frame = cap.read()
    if cap is None or not cap.isOpened():
    	print('Warning: unable to open video source: ', source)
    size = frame.shape
    cv2.imshow('frame',frame)
    #if count%30==0:
    	#cv2.imwrite("/tf/data/saved_images/"+str(count) + ".jpg", frame)
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
    print(camera_matrix)
    #if count%20==0:
    #    print("Capturing")
        #cv2.imwrite("captured_images/stream2"+str(count)+".jpg",frame)
    #print(camera_matrix)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
