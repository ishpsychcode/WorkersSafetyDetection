import os
import pickle
import numpy as np
from deepface import DeepFace

# -------------------------
# SETTINGS
# -------------------------
BASE_DIR = "facial features"
ENCODING_PATH = "encodings.pickle"
MODEL = "Facenet512"
DETECTOR = "retinaface"

def generate_encodings():

    known_data = []

    if not os.path.exists(BASE_DIR):
        print(f"❌ Folder {BASE_DIR} not found")
        return

    for person_name in os.listdir(BASE_DIR):

        person_folder = os.path.join(BASE_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        print(f"\n🔍 Processing {person_name}")

        for image_name in os.listdir(person_folder):

            img_path = os.path.join(person_folder, image_name)

            try:

                embeddings = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL,
                    detector_backend=DETECTOR,
                    enforce_detection=True
                )

                for emb in embeddings:

                    embedding = np.array(emb["embedding"])

                    embedding = embedding / np.linalg.norm(embedding)

                    known_data.append({
                        "name": person_name,
                        "embedding": embedding
                    })

                    print(f"   ✅ Added {image_name}")

            except Exception as e:
                print(f"   ⚠️ Skipping {image_name}")

    with open(ENCODING_PATH, "wb") as f:
        pickle.dump(known_data, f)

    print("\n==============================")
    print("✅ Training complete")
    print(f"Saved to {ENCODING_PATH}")
    print("==============================")

if __name__ == "__main__":
    generate_encodings()