import numpy as np
import cv2
import sys

video_path = 'D:/Projects/HubinoWFH/Army Mog Object Detection/pavement.mp4'
cv2.ocl.setUseOpenCL(False)


cap = cv2.VideoCapture(video_path)

fgbg = cv2.createBackgroundSubtractorMOG2()
	

while (cap.isOpened):

	
	ret, frame = cap.read()
	
	if ret==True:

		#apply background substraction
		fgmask = fgbg.apply(frame)
					
		
		(contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		
		
		for c in contours:
			if cv2.contourArea(c) < 500:
				continue
				
			
			(x, y, w, h) = cv2.boundingRect(c)
			
			
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			
		cv2.imshow('foreground and background',fgmask)
		cv2.imshow('rgb',frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break


cap.release()
cv2.destroyAllWindows()