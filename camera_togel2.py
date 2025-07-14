import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime
import stat
from ultralytics import YOLO    

# Ensure the YOLO model is loaded correctly
# Update the path to your YOLO model as needed  
yolo_model = YOLO(r"D:\facerecognition\models\best.pt")  # Replace with your YOLO model path

def load_known_faces(photos_dir="photos"):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(photos_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(photos_dir, filename)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
    return known_face_encodings, known_face_names

def ensure_attendance_file_exists(filename):
    if os.path.exists(filename):
        os.chmod(filename, stat.S_IWRITE)

    # If not exists, create and add header
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Timestamp"])

def main():
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces()
    
    # Attendance setup
    attendance_file = "attendance.csv"
    ensure_attendance_file_exists(attendance_file)
    attendance_marked = set()

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Camera not accessible.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and get encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Mark attendance if not already marked
            if name != "Unknown" and name not in attendance_marked:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, now])
                print(f"Marked attendance for {name} at {now}")
                attendance_marked.add(name)

            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom + 5), (right, bottom + 25), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Show the frame
        cv2.imshow('Webcam Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
