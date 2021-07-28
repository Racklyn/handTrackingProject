from typing import Sequence
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() #creating a new object Hand
mpDraw = mp.solutions.drawing_utils

pTime = 0 #previous
cTime = 0 #current


while True:
    success, img = cap.read()
    # convertendo de BGR para RGB, no padrão que é usado por Hands
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB) #Aqui a mágina acontece

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark): #percorrendo cara um dos 21 pontos de cada mão
                # print(id, lm)
                h,w,c = img.shape # height, weight, channels
                cx, cy = int(lm.x*w), int(lm.y*h) # posição do centro de cada ponto (landmark)
                print(id, cx, cy)
                # cv2.rectangle(img, [cx,cy], [cx+10,cy+10], (255,0,255), 1)
                if id==4:
                    cv2.circle(img, (cx, cy), 15, (0, 200, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps))+" fps", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

    cv2.imshow("Image", img) #cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    cv2.waitKey(1)
