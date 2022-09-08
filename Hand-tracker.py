import cv2
import mediapipe as mp
import time

# => turn on our webcam
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

# => draw line between each distance of point for each hand
mpDRAW =mp.solutions.drawing_utils


# => create fps time

pTime = 0
cTime = 0


# => in this block we run our webcam
while True:
    success, img = cap.read()
    # => MEDIAPIPE OBJECT JUST USE RGB
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # => process frame
    results = hands.process(img)
    #print(results.multi_hand_landmarks)


    # => get information form eacch hand and draw hand
    if results.multi_hand_landmarks:
        for handlandmarks in results.multi_hand_landmarks:
            # => get exact information of index finger landmark
            for id , landmarks in enumerate(handlandmarks.landmark):
                h , w, c = img.shape
                # = > find position
                cx , cy = int(landmarks.x*w)  , int(landmarks.y*h)
                print(id,cx,cy)

                cv2.circle(img,(cx,cy),7,(0, 255, 255),cv2.FILLED)


            mpDRAW.draw_landmarks(img , handlandmarks , mpHands.HAND_CONNECTIONS)

    # => fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # => show position of frame rate on window
    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,255),3)


    cv2.imshow('image',img)
    cv2.waitKey(1)
