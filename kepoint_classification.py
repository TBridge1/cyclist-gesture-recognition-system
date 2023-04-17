# imports for training model
import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

# paths
dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'
#####
# needs to be set to number of classes 4 in this case for neutral, stop, direction and thanks
NUM_CLASSES = 4
#####
# gets dataset variables
# x is co ordinates
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
# y is the gesture indentifier
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
# splits train test into 0.75 to 0.25
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
# This creates a new Sequential model object from the Keras API.
model = tf.keras.models.Sequential([
    # This adds an input layer to the model with a shape of (21 * 2,), which means it expects an input tensor of shape
    # (batch_size, 42). This layer doesn't perform any computation; it just defines the shape of the input data.
    # 21 refers to the 21 landmarks and 42 as there are two points for each one
    tf.keras.layers.Input((21 * 2,)),
    # This adds a dropout layer to the model with a dropout rate of 0.2,
    # which means that during training, 20% of the input units will be randomly
    # set to 0 at each update, to prevent overfitting.
    tf.keras.layers.Dropout(0.2),
    # This adds a fully connected layer with 20 units and ReLU activation function.
    tf.keras.layers.Dense(20, activation='relu'),
    # This adds another dropout layer with a dropout rate of 0.4.
    tf.keras.layers.Dropout(0.4),
    # This adds another fully connected layer with 10 units and ReLU activation function.
    tf.keras.layers.Dense(10, activation='relu'),
    # This adds the output layer with a number of units equal to the number of classes in the problem
    # (represented by the NUM_CLASSES variable in this case 4)
    # and a softmax activation function, which outputs a probability distribution over the classes.
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)
# Model checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
# Callback for early stopping
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
# Model compilation
# configures the model to use the Adam optimizer, sparse categorical cross-entropy loss,
# and accuracy as the evaluation metric during training and testing.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# fits the data to the created model and sets epochs and batch size etc
model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)

# Model evaluation
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
# Loading the saved model
model = tf.keras.models.load_model(model_save_path)

# Inference test
predict_result = model.predict(np.array([X_test[0]]))
print(np.squeeze(predict_result))
print(np.argmax(np.squeeze(predict_result)))

# graphing imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


# confusion matrix function for 4 classes
def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    print(labels)
    labels1 = ["Neutral", "Stop", "Direction", "Thanks"]
    # confusion matric based on true values vs predicted
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    # creates df for graphing with labels of classes
    df_cmx = pd.DataFrame(cmx_data, index=labels1, columns=labels1)

    fig, ax = plt.subplots(figsize=(7, 6))
    # seaborn heatmap
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))


# creates predicted values
Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

# calls cm function
print_confusion_matrix(y_test, y_pred)

# Save as a model dedicated to inference
model.save(model_save_path, include_optimizer=False)

# Transform model (quantization)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

open(tflite_save_path, 'wb').write(tflite_quantized_model)

interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()
# Get I / O tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))
# %%time
# Inference implementation
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])
print(np.squeeze(tflite_results))
print(np.argmax(np.squeeze(tflite_results)))
