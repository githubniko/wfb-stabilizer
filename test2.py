#!/usr/bin/python3
import cv2
import numpy
import datetime

PORT = '5601'

SRC = f'udpsrc port={PORT} caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H265, payload=(int)96" \
! rtph265depay ! h265parse ! mppvideodec ! videoparse width=1280 height=720 format=nv12 framerate=30/1 ! videoconvert \
! appsink sync=false'



cap = cv2.VideoCapture(SRC);
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print('Src opened, %dx%d @ %d fps' % (w, h, fps))

file = f'record_{PORT}_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}.mp4'


pipeline = f"appsrc  ! queue ! videoconvert ! video/x-raw,format=NV12 ! mpph265enc ! filesink location={file}";
#pipeline = f"appsrc  ! videoconvert !  xvimagesink sync=false"; # вывод через gstreamer

out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)))

if not cap.isOpened():
    print("Cannot capture test src. Exiting.")
    quit()

window_name=f'OpenIPC'
cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)    

buff = numpy.zeros(shape=(int(h), int(w), 3), dtype=numpy.uint8)

while True:
    ret, frame = cap.read(buff)
    if ret == False:
        break
	
    out.write(frame)
 #   cv2.imshow(window_name,frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()
