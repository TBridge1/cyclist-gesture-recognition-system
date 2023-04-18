
import cv2
import numpy as np
from google.protobuf.json_format import MessageToDict

from tensorflow.keras.models import Model, load_model

import mediapipe as mp


mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)
mpDraw = mp.solutions.drawing_utils

model = load_model("rps4.h5")

# This list will be used to map probabilities to class names, Label names are in alphabetical order.
label_names = ['nothing', 'paper', 'rock', 'scissor']

cap = cv2.VideoCapture(0)
box_size = 234
width = int(cap.get(3))

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    x, y, c = frame.shape

    cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 250, 150), 2)

    cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_NORMAL)

    roi = frame[5: box_size - 5, width - box_size + 5: width - 5]

    # Normalize the image like we did in the preprocessing step, also convert float64 array.
    roi = np.array([roi]).astype('float64') / 255.0

    # Get model's prediction.
    pred = model.predict(roi)

    # Get the index of the target class.
    target_index = np.argmax(pred[0])

    # Get the probability of the target class
    prob = np.max(pred[0])

    results = hands.process(frame)

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
            cv2.putText(frame, 'Both Hands', (250, 100),
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
                                (20, 100),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)

                if label == 'Right':
                    # Display 'Left Hand'
                    # on left side of window
                    cv2.putText(frame, label + ' Hand', (460, 100),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.9, (0, 255, 0), 2)

    # Show results
    cv2.putText(frame, "prediction: {} {:.2f}%".format(label_names[np.argmax(pred[0])], prob * 100),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Rock Paper Scissors", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()