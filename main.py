import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Set threshold for high confidence match (tune if needed)
THRESHOLD = 0.80  # cosine similarity

# Use GPU if available (0 = first GPU, -1 = CPU)
CTX_ID = 0  # change to -1 if running on CPU only

# Load the InsightFace model (buffalo_l includes ArcFace, RetinaFace, gender/age)
print("Loading model...")
model = FaceAnalysis(name = "buffalo_l")
model.prepare(ctx_id = CTX_ID)

# Load two images
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img

# Get face embedding
def get_embedding(img):
    faces = model.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    return faces[0].embedding

# Cosine similarity between two embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare two face images
def compare_faces(img_path1, img_path2):
    img1 = load_image(img_path1)
    img2 = load_image(img_path2)

    print("Extracting embeddings...")
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    similarity = cosine_similarity(emb1, emb2)
    confidence = round(((similarity + 1) / 2) * 100, 2)  # optional

    print(f"\nCosine Similarity: {similarity:.4f}")
    print(f"Confidence Score: {confidence:.2f}%")

    if similarity >= THRESHOLD:
        print("MATCH: High confidence the same person.")
    else:
        print("NO MATCH: Likely different people.")

if __name__ == "__main__":
    path_to_image1 = "face1.jpg"
    path_to_image2 = "face2.jpg"

    compare_faces(path_to_image1, path_to_image2)
