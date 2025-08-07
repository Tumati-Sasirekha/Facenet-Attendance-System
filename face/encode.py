from keras_facenet import FaceNet  # type: ignore
import os
import pickle

embedder = FaceNet()

DATASET_DIR = DATASET_DIR = 'dataset'
ENCODING_FILE = 'face_encodings.pkl'       


known_encodings = []
known_names = []

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        print(f"[INFO] Processing {img_path}")

        detections = embedder.extract(img_path)

        if detections:
            embedding = detections[0]["embedding"]
            known_encodings.append(embedding)
            known_names.append(person_name)
        else:
            print(f"[WARNING] No face detected in {img_path}")

data = {"encodings": known_encodings, "names": known_names}
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Saved {len(known_names)} face embeddings to {ENCODING_FILE}")
