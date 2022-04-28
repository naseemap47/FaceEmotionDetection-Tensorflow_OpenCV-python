import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np

face_classifier = cv2.CascadeClassifier(
    '/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python/haarcascade_frontalface_default.xml'
)
model = load_model('/home/naseem/PycharmProjects/FaceEmotionDetection-Tensorflow_OpenCV-python/Model.h5')

emotion_labels = [
    'Angry', 'Disgust', 'Fear',
    'Happy', 'Neutral', 'Sad',
    'Surprise'
]

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(img_rgb)
    # print(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = img_rgb[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_AREA)
        # print(len(face_roi))

        if len(face_roi) != 0:
            roi = face_roi.astype('float32') / 255
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)[0]
            # print(prediction)
            predict_emotion = emotion_labels[prediction.argmax()]
            # print(predict_emotion)
            cv2.putText(
                img, str(predict_emotion), (x + 5, y - 5),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2
            )

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
