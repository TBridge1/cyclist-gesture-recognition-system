# Using Gesture Recognition to Show a Cyclist's Intent
Estimate hand pose using MediaPipe in Python 3.10.<br> This program adapts a [GitHub](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
and translated to [English](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) to create a series of gestures
a cyclist can use to show their intents to other road users.

The four classes are Neutral, Stop, Direction and Thanks.
An Arduino with LEDs connected through Serial Port and a webcam is required.
An external webcam is better suited to get the right angle for viewing hands.

# Requirements
* Python 3.10
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix) 
* matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)


# Demo
Here's how to run the demo using your webcam.
```bash
python app.py
```

The following options can be specified when running the demo.
* --device<br>Specifying the camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：960)
* --height<br>Height at the time of camera capture (Default：540)
* --use_static_image_mode<br>Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
* --min_detection_confidence<br>
Detection confidence threshold (Default：0.5)
* --min_tracking_confidence<br>
Tracking confidence threshold (Default：0.5)

# Directory
<pre>
│  app.py
│  keypoint_classification.py
│
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│
├───ArduinoCode
│
├───Prototype Work
│   └───CV
│       ├───handdetect.py
│       ├───rps4.h5
│       ├───rpscam2.py
│       ├───rpscamera.py
│       ├───rpsgather.py
││      ├───rpsmodel.h5
│       ├───rpsmodel.py
│       ├───rpstest.py
│       │
│       ├───hand-gesture-recognition-code
│       │   └───mp_hand_gesture
│       │       │───hand_gesture_detection.py
│       └───rps
│           └───Rock-Paper-Scissors
│               ├───Rock-Paper-Scissors
│               │   ├───test
│               │   │   ├───paper
│               │   │   ├───rock
│               │   │   └───scissors
│               │   ├───train
│               │   │   ├───paper
│               │   │   ├───rock
│               │   │   └───scissors
│               │   └───validation
│               ├───test
│               │   ├───paper
│               │   ├───rock
│               │   └───scissors
│               ├───train
│               │   ├───paper
│               │   ├───rock
│               │   └───scissors
│               └───validation
│
│        
└─utils
    └─cvfpscalc.py
</pre>

### app.py
This is the main program for interacting and recognising gestures.<br>
In addition, learning data (key points) for hand sign recognition,<br>

### keypoint_classification.py
This is a model training script for hand sign recognition.


### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)


### utils/cvfpscalc.py
This is a module for FPS measurement.

### Prototype Work
Contains all the various programs and exploits made in the creation of the project
there is a lot of overlap with the work done in the main app.py and keypoint_classifier.py file.
In there there is:
* OpenCV camera Intgration
* Hand and Handedness Detection
* Model Training
* Gathering of Data
* Augmentation of Data
* Graph Generation
* MediaPipe Integration with pre trained model

To re iterate the fact that this is built off others work but I had explored
most of these options before finding the GitHub it just combined everything together
and had it work with MediaPipe which was difficult to figure out without it.

# Training
Hand sign recognition and can add and change training data and retrain the model.

### Hand sign recognition training
#### 1.Learning data collection
Run "[app.py](app.py)" and
press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）<br>
If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates<br>

#### 2.Model training
Open "[keypoint_classification.py](keypoint_classification.py)" is responsible for training the model.
Running the file will take the collected data in the CSV and pass it through the Keras model created. It will also
display a Confusion Matrix and Classification Report of the model to understand how successful it is.<br>

#### 3.Arduino Compatibility
Once a model has been trained the Gestures should be recognised with a high degree of accuracy, once a gesture has been verfiied
it will display the appropriate colour LED on the Arduino breadboard.

# Reference
* [MediaPipe](https://mediapipe.dev/)

# Author
Thomas Bridgeman

# Adapted from Author/Translator
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)
Nikita Kiselov(https://github.com/kinivi)
 
# License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).
