import time
import cv2
import mediapipe
 
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
 
capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)
ptime=0
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=3) as hands:
 
    while (True):
 
        ret, frame = capture.read()
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(frame,f'FPS {int(fps)}',(40,70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 3)
        
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
 
        if results.multi_hand_landmarks != None:
            for handLandmarks in results.multi_hand_landmarks:
                drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)
 
        cv2.imshow('Test hand', frame)
 
        if cv2.waitKey(1) == 27:
            break
 
cv2.destroyAllWindows()
capture.release()