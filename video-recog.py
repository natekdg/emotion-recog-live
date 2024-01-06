# beginning creation of live image recognition through webcam

import cv2		# used for image/video processing
from deepface import DeepFace		# used for facial analysis "deepface"

# uses pre-trained emotion recognition library from deepface
model = DeepFace.build_model("Emotion")


