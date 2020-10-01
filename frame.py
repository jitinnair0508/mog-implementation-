import cv2

# Opens the Video file
cap= cv2.VideoCapture('D:/Projects/HubinoWFH/Army Mog Object Detection/2.MTS')
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('kang'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()