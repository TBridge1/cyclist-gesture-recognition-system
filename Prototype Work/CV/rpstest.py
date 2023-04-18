from keras import models
#the imports for creating a model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
model = models.load_model('rpsmodel.h5')
test_dir = 'rps/Rock-Paper-Scissors/test'
test_generator = ImageDataGenerator(rescale = 1./255,
                                   validation_split = 0.7)

test_images = test_generator.flow_from_directory(test_dir,
                                                target_size = (150, 150),
                                                color_mode='rgb',
                                                class_mode='categorical',
                                                batch_size=32,
                                                shuffle=False,
                                                seed=42,
                                                subset='validation')

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