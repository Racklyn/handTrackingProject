import cv2
import time
import os
import handTrackingModule as htModule

wCan, hCan = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCan)
cap.set(4, hCan)

folderPath = 'fingerImages'
myList = os.listdir(folderPath) #listdir: retorna uma lista com o nome(str) dos arquivos de um dir
myList.sort()

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}') #...caminho relativo de cada imagem
    overlayList.append(image)



pTime = 0

detector = htModule.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20] # Ids das pontas dos dedos

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        fingers = []

        # Polegar:
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] + 20:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 dedos
        for id in range(1,5):
            #se y do ponto tipId("ponta do dedo") é menor que y do ponto tipId-2 ("começo do dedo")
            if  lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]: 
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1) # quantas vezes '1' aparece na lista
        # print(totalFingers)


        # mudando a imagem:
        h, w, c = overlayList[totalFingers-1].shape # altura, largura e canais
        img[0:h, 0:w] = overlayList[totalFingers-1] #inserindo imagem dentro de outra

        cv2.rectangle(img, (20, 225), (170, 425), (0, 180, 255), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 25)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (450, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)