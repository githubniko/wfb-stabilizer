#!/usr/bin/python3
# Author: ejowerks
# Version 0.00000000001 Proof of Concept Released 4/3/2023
# Open Source -- Do what you wanna do
# Thanks to https://github.com/trongphuongpro/videostabilizer 

import cv2
import numpy as np
import sys
import datetime

# Usage: python ejo_wfb_stabilizer.py [optional video file]
# press "Q" to quit

#################### USER VARS ######################################
# Decreases stabilization latency at the expense of accuracy. Set to 1 if no downsamping is desired. 
# Example: downSample = 0.5 is half resolution and runs faster but gets jittery
downSample = 1.0

#Zoom in so you don't see the frame bouncing around. zoomFactor = 1 for no zoom
zoomFactor = 0.9

# pV and mV can be increased for more smoothing #### start with pV = 0.01 and mV = 2 
processVar=0.03
measVar=2

# set to 1 to display full screen -- doesn't actually go full screen if your monitor rez is higher than stream rez which it probably is. TODO: monitor resolution detection
showFullScreen = 0

# If test video plays too fast then increase this until it looks close enough. Varies with hardware. 
# LEAVE AT 1 if streaming live video from WFB (unless you like a delay in your stream for some weird reason)
delay_time = 1 


######################## Region of Interest (ROI) ###############################
# This is the portion of the frame actually being processed. Smaller ROI = faster processing = less latency
#
# roiDiv = ROI size divisor. Minimum functional divisor is about 3.0 at 720p input. 4.0 is best for solid stabilization.
# Higher FPS and lower resolution can go higher in ROI (and probably should)
# Set showrectROI and/or showUnstabilized vars to = 1 to see the area being processed. On slower PC's 3 might be required if 720p input
roiDiv = 3.5

# set to 1 to show the ROI rectangle 
showrectROI = 0

#showTrackingPoints # show tracking points found in frame. Useful to turn this on for troubleshooting or just for funzies. 
showTrackingPoints = 0

# set to 1 to show unstabilized B&W ROI in a window
showUnstabilized = 0

# maskFrame # Wide angle camera with stabilization warps things at extreme edges of frame. This helps mask them without zoom. 
# Feels more like a windshield. Set to 0 to disable or find the variable down in the code to adjust size of mask
maskFrame = 0

######################## Video Source ###############################

# Устанавливаем порт источника
if len(sys.argv) == 2 and sys.argv[1].isdigit() == True :
	PORT = sys.argv[1]	
else:
	PORT = '5601'
		
# Your stream source. Requires gstreamer libraries 
# Can be local or another source like a GS RPi
# Check the docs for your wifibroadcast variant and/or the Googles to figure out what to do. 

# Below should work on most PC's with gstreamer  -- ###  #### #### Without hardware acceleration you may need to reduce your stream to 920x540 ish #### #### ###
#SRC = 'udpsrc port=5600 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink sync=false'

# Обработка видео-файла
#SRC = 'filesrc location=720p60.mp4 ! decodebin ! videoparse width=1280 height=720 format=nv12 framerate=60/1 ! videoconvert ! appsink sync=false'

# Источник WFB для Orange Pi 5, Rockchip RK3588
SRC = f'udpsrc port={PORT} caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H265, payload=(int)96" \
! rtph265depay !  h265parse ! mppvideodec  ! videoparse width=1280 height=720 format=nv12 framerate=30/1 ! videoconvert \
! appsink sync=false'

# Below is for author's Ubuntu PC with nvidia/cuda stuff running WFB-NG locally (no groundstation RPi). Requires a lot of fiddling around compiling opencv w/ cuda support
#SRC = 'udpsrc port=5600 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay !  h264parse ! nvh264dec ! videoconvert ! appsink sync=false'



######################################################################

SAVE = False
if len(sys.argv) > 1:
	for i in range(len(sys.argv)):
		match sys.argv[i].split(): 
			case["save"]:
				SAVE = True
			case["--file"]:
				SRC = f'filesrc location={sys.argv[i+1]} ! decodebin ! videoparse width=1280 height=720 format=nv12 framerate=30/1 ! videoconvert ! appsink sync=false'

lk_params = dict( winSize  = (15,15),maxLevel = 3,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
count = 0
a = 0
x = 0
y = 0
Q = np.array([[processVar]*3])
R = np.array([[measVar]*3])
K_collect = []
P_collect = []
prevFrame = None

video = cv2.VideoCapture(SRC)
w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)
print('Src opened, %dx%d @ %d fps' % (w, h, fps))
	
# Инициализация сохранения в файл
if SAVE:	
	file = f'record_{PORT}_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}.mkv'
	pipeline = f"appsrc  ! queue ! videoconvert ! video/x-raw,format=NV12 ! mpph265enc ! filesink location={file}";
	#pipeline = f"appsrc  ! videoconvert !  xvimagesink sync=false"; # вывод через gstreamer
	out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, float(fps), (int(w), int(h)))

buff = np.zeros(shape=(int(h), int(w), 3), dtype=np.uint8) # создаем буффер, чтобы программа не создавала его каждый раз https://stackoverflow.com/a/53540699/17709673

while True:
	grab, frame = video.read(buff)
	if grab is not True:
		exit() 
	res_w_orig = frame.shape[1]
	res_h_orig = frame.shape[0]
	
	# Расчет прямоугольной области для вычислений
	res_w = int(res_w_orig * downSample)
	res_h = int(res_h_orig * downSample)
	top_left= [int(res_h/roiDiv),int(res_w/roiDiv)]
	bottom_right = [int(res_h - (res_h/roiDiv)),int(res_w - (res_w/roiDiv))]
	frameSize=(res_w,res_h)
	
	Orig = frame
	
	# Обезаниае кадра, если установлен размер обрезки
	if downSample != 1:
		frame = cv2.resize(frame, frameSize) # downSample if applicable
	
	# Преобразование в черно-белый	
	currFrame = frame
	currGray = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)
	currGray = currGray[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]  ] #select ROI

	# Если это первый кадр
	if prevFrame is None:
		prevOrig = frame
		prevFrame = frame
		prevGray = currGray
	
	# Если есть grab, то обрабатываем
	if (grab == True) & (prevFrame is not None):
		# Ресуем прямоугольник рентабильности для отладки
		if showrectROI == 1:
			cv2.rectangle(prevOrig,(top_left[1],top_left[0]),(bottom_right[1],bottom_right[0]),color = (211,211,211),thickness = 1)
		
		# Not in use, save for later
		#gfftmask = np.zeros_like(currGray)
		#gfftmask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 255
		
		# Определение угла изображения
		prevPts = cv2.goodFeaturesToTrack(prevGray,maxCorners=400,qualityLevel=0.01,minDistance=30,blockSize=3)
		if prevPts is not None:
			# Отслеживание характерных точек
			currPts, status, err = cv2.calcOpticalFlowPyrLK(prevGray,currGray,prevPts,None,**lk_params)
			assert prevPts.shape == currPts.shape
			idx = np.where(status == 1)[0]
			# Добавьте исходное разрешение видео в пунктах к показателю рентабельности инвестиций в пунктах
			prevPts = prevPts[idx] + np.array([int(res_w_orig/roiDiv),int(res_h_orig/roiDiv)]) 
			currPts = currPts[idx] + np.array([int(res_w_orig/roiDiv),int(res_h_orig/roiDiv)])
			
			# Рисуем точки-ориентиры
			if showTrackingPoints == 1:
				for pT in prevPts:
					cv2.circle(prevOrig, (int(pT[0][0]),int(pT[0][1])) ,5,(211,211,211))
			
			# Ищем следующие преобразования: поворот, масштабирование и перенос (4 степени свободы).
			if prevPts.size & currPts.size:
				m, inliers = cv2.estimateAffinePartial2D(prevPts, currPts)
			
			if m is None:
				m = lastRigidTransform
				
			# Сглаживающий
			dx = m[0, 2] # смещение по х
			dy = m[1, 2] # смещение по у
			da = np.arctan2(m[1, 0], m[0, 0]) # угол поворота
		else:
			dx = 0
			dy = 0
			da = 0

		# Переводим в абсолютные значения
		x += dx
		y += dy
		a += da
		Z = np.array([[x, y, a]], dtype="float")
		
		# Магические вычичления
		if count == 0:
			X_estimate = np.zeros((1,3), dtype="float")
			P_estimate = np.ones((1,3), dtype="float")
		else:
			X_predict = X_estimate
			P_predict = P_estimate + Q
			K = P_predict / (P_predict + R)
			X_estimate = X_predict + K * (Z - X_predict)
			P_estimate = (np.ones((1,3), dtype="float") - K) * P_predict
			K_collect.append(K)
			P_collect.append(P_estimate)
			
		#
		diff_x = X_estimate[0,0] - x
		diff_y = X_estimate[0,1] - y
		diff_a = X_estimate[0,2] - a
		dx += diff_x
		dy += diff_y
		da += diff_a
		m = np.zeros((2,3), dtype="float")
		m[0,0] = np.cos(da)
		m[0,1] = -np.sin(da)
		m[1,0] = np.sin(da)
		m[1,1] = np.cos(da)
		m[0,2] = dx
		m[1,2] = dy

		# Приминение к основному кадру
		fS = cv2.warpAffine(prevOrig, m, (res_w_orig,res_h_orig)) # apply magic stabilizer sauce to frame
		s = fS.shape
		T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, zoomFactor)
		f_stabilized = cv2.warpAffine(fS, T, (s[1], s[0]))
		
		window_name=f'Stabilized:{res_w}x{res_h}'
		cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
		
		# Что-то связанное с маской кадра ???
		if maskFrame == 1:
			mask = np.zeros(f_stabilized.shape[:2], dtype="uint8")
			cv2.rectangle(mask, (100, 200), (1180, 620), 255, -1)
			f_stabilized = cv2.bitwise_and(f_stabilized, f_stabilized, mask=mask)
		
		# Если задан парамтер, то разворачиваем на весь экран
		if showFullScreen == 1:
			cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
		
		# Отображение кадра
		cv2.imshow(window_name, f_stabilized)
		
		# Сохранение в файл
		if SAVE:
			out.write(f_stabilized)
		
		# Если параметр установлен, то отображается 2ое окно с областью рантабельности ROI
		if showUnstabilized == 1:
			cv2.imshow("Unstabilized ROI",prevGray)
			
		# Обработка клавиш для закрытия приложения	
		if cv2.waitKey(delay_time) & 0xFF == ord('q'):
			break
		
		# Сохраняе кадр как предыдущий
		prevOrig = Orig
		prevGray = currGray
		prevFrame = currFrame
		lastRigidTransform = m
		count += 1
	else:
		exit()
 
video.release()
out.release()
cv2.destroyAllWindows()
