import webbrowser
import cv2
import numpy as np
#import pyzbar.pyzbar as pyzbar
import urllib.request
 
#cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
 
url='http://192.168.4.3/'
cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
 
while True:
    #webbrowser.open(url+'1024x768.mjpeg',new=1)
    img_resp=urllib.request.urlopen(url+'1024x768.jpg')
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgnp,-1)
    #_, frame = cap.read()
 
    #decodedObjects = pyzbar.decode(frame)
    
    cv2.imshow("live transmission", frame)
 
    key = cv2.waitKey(1)
    if key == 27:
        break
 
cv2.destroyAllWindows()