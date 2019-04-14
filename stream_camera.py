import numpy as np
import cv2

cap = cv2.VideoCapture('rtsp://admin:admin@192.168.214.40/h264.sdp?res=half&x0=0&y0=0&x1=1920&y1=1080&qp=16&doublescan=0&ssn=41645')

count = 0
while(True):
    count = count + 1
    ret, frame = cap.read()
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
    if count%20==0:
        print("Capturing")
        cv2.imwrite("captured_images/stream2"+str(count)+".jpg",frame)
    #print(camera_matrix)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
