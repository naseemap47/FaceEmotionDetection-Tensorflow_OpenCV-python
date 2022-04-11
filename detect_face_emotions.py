import cv2
from keras.models import load_model

cap = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier('/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python/haarcascade_frontalface_default.xml')
model = load_model('/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python/Model.h5')

emotion_labels = [
    'Angry', 'Disgust', 'Fear',
    'Happy', 'Neutral', 'Sad',
    'Surprise'
]

while True:
    success, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    # print(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)