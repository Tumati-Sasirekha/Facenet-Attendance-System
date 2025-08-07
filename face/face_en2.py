import cv2 # type: ignore
import pickle
import numpy as np # type: ignore
from keras_facenet import FaceNet # type: ignore
from scipy.spatial.distance import cosine # type: ignore

embedder = FaceNet()

with open("face_encodings.pkl", "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_ids = data["names"]  

THRESHOLD = 0.5 

cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections = embedder.extract(rgb_frame, threshold=0.95)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        embedding = det["embedding"]

        student_id = "Unknown"
        min_dist = 1 

        for known_embedding, known_id in zip(known_encodings, known_ids):
            dist = cosine(embedding, known_embedding)
            if dist < min_dist:
                min_dist = dist
                student_id = known_id

        if min_dist > THRESHOLD:
            student_id = "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{student_id} ({min_dist:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Recognition - Student ID", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()