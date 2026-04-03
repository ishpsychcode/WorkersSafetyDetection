import os
import pickle
import numpy as np
from deepface import DeepFace

BASE_DIR = "facial features"
ENCODING_PATH = "encodings.pickle"

def generate_encodings():
    known_data = []

    for person_name in os.listdir(BASE_DIR):
        person_folder = os.path.join(BASE_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        print(f"Processing: {person_name}")

        for image_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, image_name)

            try:
                embedding_objs = DeepFace.represent(
                    img_path,
                    model_name='Facenet512',
                    detector_backend='retinaface'  # 🔥 important
                )

                if len(embedding_objs) > 0:
                    embedding = np.array(embedding_objs[0]["embedding"])
                    
                    # 🔥 normalize
                    embedding = embedding / np.linalg.norm(embedding)

                    known_data.append({
                        "name": person_name,
                        "embedding": embedding
                    })

            except Exception as e:
                print(f"Skipping {image_name}: {e}")

    with open(ENCODING_PATH, "wb") as f:
        pickle.dump(known_data, f)

    print("✅ Training complete!")
    

if __name__ == "__main__":
    generate_encodings()
