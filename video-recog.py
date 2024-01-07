# beginning creation of live image recognition through webcam

import cv2		# used for image/video processing
from deepface import DeepFace		# used for facial analysis "deepface"

# uses pre-trained emotion recognition library from deepface
model = DeepFace.build_model("Emotion")

# create the labels for emotion
emotion_labels = ['disgust', 'fear', 'sad', 'happy', 'angry', 'suprise', 'neutral']

# import the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# pull up video from webcam
cap = cv2.VideoCapture(0)

while True:
    # capture video by each frame
    ret, frame = cap.read()

    # make the frame grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face from the frames
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # find the face ROI (region of interest)
        face_roi = gray_frame[y:y +h, x:x + w]

        # resize face ROI to match the model shape
        resized_face = cv2.resize(face_roi, (48,48), interpolation=cv2.INTER_AREA)

        # normalize face image (resized)
        normalized_face = normalized_face.reshape(1, 48, 48, 1)

        # read(predict) emotions with pre-trained model
        predict_face = model.predict(resized_face)[0]
        emotion_idx = predict_face.argmax()
        emotion = emotion_labels[emotion_idx]

        # place rectangle around face with label of emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # display the frame
    cv2.imshow('Face Emotion Detection', frame)

    # quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # stop the webcam capture and close windows
    cap.release()
    cv2.destroyAllWindows()

