import cv2
import mediapipe as mp
import time
import handTrackingModule as htModule

pTime = 0 #previous
cTime = 0 #current
cap = cv2.VideoCapture(0)
# New object:
detector = htModule.handDetector()


while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    detector.findPosition(img, 1)
    
    if len(lmList)!= 0:
        print(lmList[4]) #ponto 4 da mão (dedão)  

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps))+" fps", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

    # Mostrando imagem:
    cv2.imshow("Image", img) #cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    cv2.waitKey(1)