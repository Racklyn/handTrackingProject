import cv2
import mediapipe as mp
import time
import numpy as np  #para trabalhar com arrays
import handTrackingModule as htModule
import math
from subprocess import call


#######################################
wCam, hCam = 640, 480
#######################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam) #mudando largura
cap.set(4, hCam) #mudando altura

pTime = 0

detector = htModule.handDetector(detectionCon=0.7)
# ...aumentando detectionCon para ter certeza que é uma mão


vol = 0
volBar = 400
# volPer = 0

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2] # polegar
        x2, y2 = lmList[8][1], lmList[8][2] # dedo indicador
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # centro da linha
                # //  floor division

        cv2.circle(img, (x1,y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx,cy), 13, (255, 0, 255), cv2.FILLED)

        #tamanho da reta (hipotenusa)
        length = math.hypot(x2 - x2, y1 - y2)

        # Hand range 50 - 300

        # convertendo proporcionalmente length
        vol = np.interp(length, [40,200], [0, 100])
        volBar = np.interp(length, [40,200], [400, 150])
        
        # Alterando volume:
        call(["amixer", "-D", "pulse", "sset", "Master", f'{vol}%'])

        if length < 40:
            cv2.circle(img, (cx,cy), 13, (0, 0, 255), cv2.FILLED)
        if length >= 200:
            cv2.circle(img, (cx,cy), 13, (0, 170, 255), cv2.FILLED)


    cv2.rectangle(img, (50, 150), (75, 400), (255, 30, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (75, 400), (255, 30, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 30, 0), 3)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)