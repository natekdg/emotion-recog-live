import cv2

from deepface import DeepFace

# bring the pre-trained model for emotion detection
model = DeepFace.build_model("Emotion")

# give labels for emotion
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# start the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# introduce webcam capture
cap = cv2.VideoCapture(0)

while True:
    # capture all frames from webcam
    ret, frame = cap.read()

    # convert frames that are captured to greyscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find faces in greyscale frames
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # extract the region of interest (roi) - the face
        face_roi = gray_frame[y:y + h, x:x + w]

        # number divisions for face
        num_divisions = 10


        # resize the face roi to match the model
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # normalize the resized face for model prediction
        normalized_face = resized_face / 255.0

        # reshape the face image for the model
        reshaped_face = normalized_face.reshape(1, 48, 48, 1)

        # use the model to predict the emotion of the face
        preds = model.predict(reshaped_face)[0]
        emotion_idx = preds.argmax()
        emotion = emotion_labels[emotion_idx]

        # draw a rectangle around the face and label it with the predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # draw the grid for the face detection
        for i in range(1, num_divisions):
            # horizontal lines
            cv2.line(frame, (x, y + i * h // num_divisions), (x + w, y + i * h // num_divisions), (0, 255, 0), 1)
            # vertical lines
            cv2.line(frame, (x + i * w // num_divisions, y), (x + i * w // num_divisions, y + h), (0, 255, 0), 1)


    # display frame with live capture
    cv2.imshow('Live Emotion Detection', frame)

    # press 'q' to exit the loop and end the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# quit all activities and webcam
cap.release()
cv2.destroyAllWindows()
