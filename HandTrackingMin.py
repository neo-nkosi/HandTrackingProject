import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

#formality for using mediapipe in the way we want to
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# go to the implementation of Hands() to view parameters
'''
parameters:

- static image mode (False)
- max no. of hands (2)
- min detection confidence(0.5)
- min tracking confidence(0.5)

'''
hands = mpHands.Hands()

# Webcam generation, continuosly captures and processes frames
while True:
    #capture
    success, img = cap.read()

    #convert to ideal format
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #process using mediapipe
    results = hands.process(imgRGB)

    #print statement to check if hands are identified in webcam (known as hand landmarks)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        #for each hand
        for handLms in results.multi_hand_landmarks:
            # print coordinates of each dot of a hand (ie each node of a hand landmark)
            # these values range from 0-1 as they are representing the ratio of their position. we must
            # (2) multiply this by the width and height of the image to get actual position values
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                #(2)
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

                #id 0 is the palm, we make the circle biggger for that node:
                if id ==0:
                    cv2.circle(img, (cx, cy), 25, (255,0,255), cv2.FILLED)
                    
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime= time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
