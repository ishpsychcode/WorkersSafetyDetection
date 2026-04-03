from deepface import DeepFace
from scipy.spatial import distance
import numpy as np

# -------------------------
# IMAGE PATHS
# -------------------------
img1 = "facial features/mrunal/WhatsApp Image 2026-03-28 at 7.41.19 PM (1).jpeg"
img2 = "facial features/ishita/IMG-20260328-WA0055.jpg"

try:
    # -------------------------
    # GENERATE EMBEDDINGS
    # -------------------------
    emb1_obj = DeepFace.represent(
        img1,
        model_name='Facenet512',
        detector_backend='retinaface'  # 🔥 IMPORTANT
    )

    emb2_obj = DeepFace.represent(
        img2,
        model_name='Facenet512',
        detector_backend='retinaface'
    )

    if len(emb1_obj) == 0 or len(emb2_obj) == 0:
        print("❌ No face detected in one of the images")
        exit()

    emb1 = emb1_obj[0]["embedding"]
    emb2 = emb2_obj[0]["embedding"]

    # -------------------------
    # NORMALIZE
    # -------------------------
    emb1 = np.array(emb1) / np.linalg.norm(emb1)
    emb2 = np.array(emb2) / np.linalg.norm(emb2)

    # -------------------------
    # DISTANCE
    # -------------------------
    dist = distance.cosine(emb1, emb2)

    print("\n✅ Distance:", dist)

    # -------------------------
    # INTERPRET RESULT
    # -------------------------
    if dist < 0.4:
        print("🟢 Same person (very confident)")
    elif dist < 0.55:
        print("🟡 Likely same person")
    else:
        print("🔴 Different person / poor data")

except Exception as e:
    print("❌ Error:", e)