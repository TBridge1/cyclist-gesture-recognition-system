import cv2
import numpy as np
from google.protobuf.json_format import MessageToDict
from keras import models
import mediapipe as mp


mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)
mpDraw = mp.solutions.drawing_utils
classNames = ["rock", "paper", "scissors"]
model = models.load_model('rpsmodel.h5')
# Start capturing Video through webcam
video = cv2.VideoCapture(0)

# runs the reading and processing of webcam frames until stopped
while True:
    _, frame = video.read()
    kernel = np.ones((3, 3), np.uint8)
    frame = cv2.flip(frame, 1)
    x, y, c = frame.shape

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    # extract skin colur image
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # extrapolate the hand to fill dark spots within
    mask = cv2.dilate(mask, kernel, iterations=4)
    # blur the image
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    mask = cv2.resize(mask, (150, 150))
    img_array = np.array(mask)
    # print(img_array.shape)
    # Changing dimension from 128x128 to 128x128x3
    img_array = np.stack((img_array,) * 3, axis=-1)
    # Our keras model used a 4D tensor, (images x height x width x channel)
    # So changing dimension 128x128x3 into 1x128x128x3
    img_array_ex = np.expand_dims(img_array, axis=0)

    # Calling the predict method on model to predict gesture in the frame

    prediction = model.predict(img_array_ex)
    #prediction = np.argmax(model.predict(img_array_ex), axis=1)
    print(prediction)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # If hands are present in image(frame)
    if results.multi_hand_landmarks:
        landmarks = []
        for handslms in results.multi_hand_landmarks:
            # loops landmarks in hands
            for lm in handslms.landmark:
                # print(id, lm)
                # adds x an y pos to list
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        # Both Hands are present in image(frame)
        if len(results.multi_handedness) == 2:
            # Display 'Both Hands' on the image
            cv2.putText(frame, 'Both Hands', (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.9, (0, 255, 0), 2)

        # If any hand present
        else:
            for i in results.multi_handedness:

                # Return whether it is Right or Left Hand
                label = MessageToDict(i)['classification'][0]['label']

                if label == 'Left':
                    # Display 'Left Hand' on
                    # left side of window
                    cv2.putText(frame, label + ' Hand',
                                (20, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)

                if label == 'Right':
                    # Display 'Left Hand'
                    # on left side of window
                    cv2.putText(frame, label + ' Hand', (460, 50),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)
                print(prediction[0][1])
                # if prediction[0] == 0:
                #      gesture = "paper"
                # elif prediction[0] == 1:
                #      gesture = "scissors"
                # else:
                #      gesture = "rock"
                if prediction[0][0] == 1:
                     gesture = "paper"
                elif prediction[0][1] == 1:
                     gesture = "scissors"
                else:
                     gesture = "rock"


                cv2.putText(frame, gesture, (250, 250), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
