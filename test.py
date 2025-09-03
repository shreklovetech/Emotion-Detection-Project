import cv2
import numpy as np
from keras.models import load_model

model = load_model(r'C:\Users\Khushi Kothari\Desktop\Emotion_Detection_CNN-main\emotion_detection_model.h5')

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
label = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

frame = cv2.imread(r"C:\Users\Khushi Kothari\Desktop\Emotion_Detection_CNN-main\test_image2.jpg")
if frame is None:
    raise FileNotFoundError("Image not found. Please check the file path.")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    roi = gray[y:y+h, x:x+w]
    resized = cv2.resize(roi, (48, 48))
    resized = resized.astype('float32') / 255.0
    reshaped = np.reshape(resized, (1, 48, 48, 1))
    preds = model.predict(reshaped)
    emotion = label[np.argmax(preds)]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

cv2.imshow('Emotion Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
