import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime
import stat
from ultralytics import YOLO

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
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Timestamp", "Role", "Helmet","Goggles","Jacket", "Gloves", "Footwear"])

def detect_safety_equipment(frame, model, class_map):
    results = model(frame)[0]
    detected = {k: False for k in class_map.values()}
    for box in results.boxes:
        class_id = int(box.cls[0])
        label = class_map.get(class_id)
        if label:
            detected[label] = True
    return detected

def main():
    # Example roles dictionary (customize as needed)
    roles = {
        "Me": "Student",
        "John": "Teacher",
        "Jane": "Student",
        "piyu": "Student"
        # Add more as needed
    }

    # Map your YOLO class indices to equipment names
    # Update these indices to match your model's class order!
    class_map = {
        0: "Helmet",
        1: "Goggles",
        2: "Jacket",
        3: "Gloves",
        4: "Footwear"
    }

    # Load YOLO model
    yolo_model = YOLO(r"D:\facerecognition\models\best.pt")  # Replace with your YOLO model path

    known_face_encodings, known_face_names = load_known_faces()
    attendance_file = "attendance.csv"
    ensure_attendance_file_exists(attendance_file)
    attendance_marked = set()

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Camera not accessible.")
        return

    print("Press 'q' to quit.")

    last_attendance = {
        "name": "", "time": "", "role": "",
        "Boots": "", "Helmet": "", "Gloves": "", "Glasses": "", "Vest": ""
    }

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Detect safety equipment
        detected = detect_safety_equipment(frame, yolo_model, class_map)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            role = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    role = roles.get(name, "Unknown")

            # Mark attendance if not already marked
            if name != "Unknown" and name not in attendance_marked:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(attendance_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        name, now, role,
                        "Yes" if detected["Boots"] else "No",
                        "Yes" if detected["Helmet"] else "No",
                        "Yes" if detected["Gloves"] else "No",
                        "Yes" if detected["Glasses"] else "No",
                        "Yes" if detected["Vest"] else "No"
                    ])
                print(f"Marked attendance for {name} at {now} ({role}), "
                      f"Boots: {'Yes' if detected['Boots'] else 'No'}, "
                      f"Helmet: {'Yes' if detected['Helmet'] else 'No'}, "
                      f"Gloves: {'Yes' if detected['Gloves'] else 'No'}, "
                      f"Glasses: {'Yes' if detected['Glasses'] else 'No'}, "
                      f"Vest: {'Yes' if detected['Vest'] else 'No'}")
                attendance_marked.add(name)
                last_attendance = {
                    "name": name,
                    "time": now,
                    "role": role,
                    "Boots": "Yes" if detected["Boots"] else "No",
                    "Helmet": "Yes" if detected["Helmet"] else "No",
                    "Gloves": "Yes" if detected["Gloves"] else "No",
                    "Glasses": "Yes" if detected["Glasses"] else "No",
                    "Vest": "Yes" if detected["Vest"] else "No"
                }

            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom + 5), (right, bottom + 25), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # --- UI/UX Overlay for attendance and safety equipment details ---
        overlay_height = 140
        overlay_color = (245, 245, 245)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), overlay_color, -1)
        cv2.putText(frame, "Last Attendance:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(
            frame,
            f"Name: {last_attendance['name']}   Time: {last_attendance['time']}   Role: {last_attendance['role']}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        cv2.putText(
            frame,
            f"Boots: {last_attendance['Boots']}   Helmet: {last_attendance['Helmet']}   Gloves: {last_attendance['Gloves']}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        cv2.putText(
            frame,
            f"Glasses: {last_attendance['Glasses']}   Vest: {last_attendance['Vest']}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
        # ------------------------------------------------------

        cv2.imshow('Webcam Face Recognition & Safety Equipment Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()