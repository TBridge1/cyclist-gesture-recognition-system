# TechVidvan hand Gesture Recognizer


##Importing of packages
# CV2 is used for the camera
# numpy is used as part of the prediction algorithm
# tensorflow is part of the deep learning model
# keras is used to read in the pre trained model
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
# the hand tracking that tracks the 21 hand points
mpHands = mp.solutions.hands
# set the amount of hands to track and a prediction must be 0.7 to give a prediction
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
# drawing the points between the hands
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names by reading a Pickle file
# oickle is way to save a pre trained model
# eg thumbs up, call me, peace etc
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam using opencv
cap = cv2.VideoCapture(0)

# runs the reading and processing of webcam frames until stopped
while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    # gets the width, height and colour of the frame
    x, y, c = frame.shape

    # Flip the frame vertically to better align gestures
    frame = cv2.flip(frame, 1)
    # opencv reads in bgr changes to rgb
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand prediction for the current frame
    result = hands.process(framergb)

    #print(result)

    className = ''

    # post process the result
    # checks if there has been hands tracked
    if result.multi_hand_landmarks:
        landmarks = []
        #loops hands
        for handslms in result.multi_hand_landmarks:
            #loops landmarks in hands
            for lm in handslms.landmark:
                # print(id, lm)
                #adds x an y pos to list
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()
