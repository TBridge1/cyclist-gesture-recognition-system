#the imports for creating a model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

#path to the train folder and the test folder
train_dir = 'CV/rps/Rock-Paper-Scissors/train'
test_dir = 'CV/rps/Rock-Paper-Scissors/test'

# sample_generator = ImageDataGenerator(rescale = 1./255,
#                                      horizontal_flip = True,
#                                      vertical_flip = True,
#                                      rotation_range = 90,
#                                      height_shift_range = 0.2,
#                                      width_shift_range = 0.2)
# sample_images = sample_generator.flow_from_directory(train_dir,
#                                                    target_size = (300, 300),
#                                                    color_mode = 'rgb',
#                                                    class_mode = None,
#                                                    batch_size = 1,
#                                                    shuffle = True,
#                                                    seed = 42)
#
# plt.figure(figsize=(10, 10))
#
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     img = sample_images.next()[0]
#     plt.imshow(img)
#     plt.axis('off')
#
# plt.show()

#augmenting the training data with flips, rescales, zoom etc
train_generator = ImageDataGenerator(rescale = 1./255,
                                    horizontal_flip = True,
                                    vertical_flip = True,
                                    rotation_range = 90,
                                    height_shift_range = 0.2,
                                    width_shift_range = 0.2,
                                    zoom_range = 0.2)
#rescaling the testing data to align with the training data
test_generator = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.7)

#create a training image shuffle
train_images = train_generator.flow_from_directory(train_dir,
                                                  target_size=(150, 150),
                                                  color_mode = 'rgb',
                                                  class_mode = 'categorical',
                                                  batch_size = 32,
                                                  shuffle = True,
                                                  seed = 42,
                                                  subset = 'training')
#shuffle the testing images
val_images = test_generator.flow_from_directory(test_dir,
                                                target_size=(150, 150),
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                batch_size=32,
                                                shuffle=True,
                                                seed=42,
                                                subset='training')
#validation image
test_images = test_generator.flow_from_directory(test_dir,
                                                target_size = (150, 150),
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                batch_size=32,
                                                shuffle=False,
                                                seed=42,
                                                subset='validation')


inputs = tf.keras.Input(shape = (150, 150 ,3))
x = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(512, activation = 'relu')(x)
outputs = tf.keras.layers.Dense(3, activation = 'softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer ='adam',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

history = model.fit(train_images,
                   validation_data = val_images,
                   epochs = 10,
                   callbacks = [
                       tf.keras.callbacks.EarlyStopping(
                       monitor = 'val_loss',
                       patience = 5,
                       restore_best_weights = True
                       )
                   ]
                   )

acc = model.evaluate(test_images, verbose=0)[1]
print("Accuracy: {:.2f}%".format(acc * 100))

predictions = np.argmax(model.predict(test_images), axis=1)


cm = confusion_matrix(test_images.labels, predictions, labels=[0, 1, 2])
clr = classification_report(test_images.labels, predictions, labels=[0, 1, 2], target_names=["Paper", "Rock", "Scissors"])

plt.figure(figsize = (10, 10))
sns.heatmap(cm, annot = True, fmt='g', vmin=0, cmap='Blues', cbar=False)
plt.xticks(ticks = [0.5, 1.5, 2.5], labels=["Paper", "Rock", "Scissors"])
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=["Paper", "Rock", "Scissors"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

model.save('rpsmodel.h5')