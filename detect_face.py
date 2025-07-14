from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
import torch
import face_recognition
import cv2
import os

# Load YOLOv7 face detector
model = attempt_load('best.pt', map_location='cpu')  # or 'cuda:0'

# Load known faces
known_face_encodings = []
known_face_names = []

for file in os.listdir('known_faces'):
    img = face_recognition.load_image_file(f'known_faces/{file}')
    encodings = face_recognition.face_encodings(img)
    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(file.split('.')[0])

# Run inference on input image
image_path = 'images/test.jpg'
img = cv2.imread(image_path)
results = model(img, size=640)[0]
results = non_max_suppression(results, conf_thres=0.5, iou_thres=0.45)

for det in results:
    if det is not None and len(det):
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            face_crop = img[y1:y2, x1:x2]
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(rgb_face)
            if encoding:
                matches = face_recognition.compare_faces(known_face_encodings, encoding[0])
                name = "Unknown"
                if True in matches:
                    name = known_face_names[matches.index(True)]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)
