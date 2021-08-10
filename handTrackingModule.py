import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        #self: atributo da classe, variável global em toda a classe
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        # Creating a new object Hand:
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):
        # convertendo de BGR para RGB, no padrão que é usado por Hands
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) #Aqui a mágina acontece

        # handsList = self.results.multi_hand_landmarks

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:   
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img  #, handsList

    def findPosition(self, img, handNo = 0, draw = True):

        lmList = [] #lista com as posições dos pontos da mão

        if self.results.multi_hand_landmarks and len(self.results.multi_hand_landmarks)>handNo: #se tiver alguma mão aparecendo (ponto da mão)
            myHand = self.results.multi_hand_landmarks[handNo] #pontos da mão em questão

            for id, lm in enumerate(myHand.landmark): #percorrendo cara um dos 21 pontos de cada mão
                    # print(id, lm)
                    h,w,c = img.shape # height, weight, channels
                    cx, cy = int(lm.x*w), int(lm.y*h) # posição do centro de cada ponto (landmark)

                    lmList.append([id, cx, cy])
                    # cv2.rectangle(img, [cx,cy], [cx+10,cy+10], (255,0,255), 1)
                    if draw:
                        cv2.circle(img, (cx, cy), 8, (0, 200, 255), cv2.FILLED)

        return lmList


def main():
    pTime = 0 #previous
    cTime = 0 #current
    cap = cv2.VideoCapture(0)
    # New object:
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList)!= 0:
            print(lmList[4]) #ponto 4 da mão (dedão)    

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps))+" fps", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

        # Mostrando imagem:
        cv2.imshow("Image", img) #cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
        cv2.waitKey(1)



# se este script/arquivo está sendo executado
if __name__ == "__main__":
    main()