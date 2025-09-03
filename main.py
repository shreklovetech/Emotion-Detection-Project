import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# Use separate variables for directories
train_dir = r"C:\Users\Khushi Kothari\Desktop\Emotion_Detection_CNN-main\images\train"
val_dir = r"C:\Users\Khushi Kothari\Desktop\Emotion_Detection_CNN-main\images\validation"

# Training hyperparameters
batch_size = 25
epochs = 150

train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                   target_size=(48, 48),
                                                   color_mode='grayscale',
                                                   class_mode='categorical',
                                                   batch_size=batch_size,
                                                   shuffle=True)

validation_generator = validation_datagen.flow_from_directory(val_dir,
                                                             target_size=(48, 48),
                                                             color_mode='grayscale',
                                                             class_mode='categorical',
                                                             batch_size=batch_size,
                                                             shuffle=False)

class_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

img, label = train_generator.__next__()

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(class_labels), activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

train_path = r"C:\Users\Khushi Kothari\Desktop\Emotion_Detection_CNN-main\images\train"
test_path = r"C:\Users\Khushi Kothari\Desktop\Emotion_Detection_CNN-main\images\validation" 
num_test_images = 0
for roots, dirs, files in os.walk(test_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            num_test_images += 1 

num_train_images = 0
for roots, dirs, files in os.walk(train_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):
            num_train_images += 1 

epochs = 150

history = model.fit(train_generator,
                    steps_per_epoch=num_train_images // batch_size,
                    validation_data=validation_generator,
                    validation_steps=num_test_images // batch_size,
                    epochs=epochs)
# Save the model
model.save(r'C:\Users\Khushi Kothari\Desktop\Emotion_Detection_CNN-main\emotion_detection_model.h5')