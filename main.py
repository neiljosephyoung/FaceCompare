import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# Get face embedding and detection info
def get_face_data(img):
    faces = model.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    return faces[0]

# Get face embedding
def get_embedding(img):
    face_data = get_face_data(img)
    return face_data.embedding

# Draw face detection box on image
def draw_face_box(img, face_data, color=(0, 255, 0), thickness=2):
    bbox = face_data.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

# Cosine similarity between two embeddings
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Visualize two images side by side with face detection boxes and connection
def visualize_comparison(img_path1, img_path2, similarity, confidence, is_match):
    img1 = load_image(img_path1)
    img2 = load_image(img_path2)
    
    face1 = get_face_data(img1)
    face2 = get_face_data(img2)
    
    # Draw face boxes
    box_color = (0, 255, 0) if is_match else (0, 0, 255)  # Green for match, red for no match
    img1_with_box = draw_face_box(img1.copy(), face1, box_color)
    img2_with_box = draw_face_box(img2.copy(), face2, box_color)
    
    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1_with_box, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_with_box, cv2.COLOR_BGR2RGB)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show images
    ax1.imshow(img1_rgb)
    ax1.set_title(f'Image 1: {img_path1}')
    ax1.axis('off')
    
    ax2.imshow(img2_rgb)
    ax2.set_title(f'Image 2: {img_path2}')
    ax2.axis('off')
    
    # Add connection line between face centers
    bbox1 = face1.bbox.astype(int)
    bbox2 = face2.bbox.astype(int)
    
    center1_x = (bbox1[0] + bbox1[2]) / 2
    center1_y = (bbox1[1] + bbox1[3]) / 2
    center2_x = (bbox2[0] + bbox2[2]) / 2
    center2_y = (bbox2[1] + bbox2[3]) / 2
    
    # Convert coordinates to figure coordinates
    fig_coord1 = ax1.transData.transform((center1_x, center1_y))
    fig_coord2 = ax2.transData.transform((center2_x, center2_y))
    
    fig_coord1 = fig.transFigure.inverted().transform(fig_coord1)
    fig_coord2 = fig.transFigure.inverted().transform(fig_coord2)
    
    # Draw connection line
    line_color = 'green' if is_match else 'red'
    line_style = '-' if is_match else '--'
    fig.add_artist(plt.Line2D([fig_coord1[0], fig_coord2[0]], 
                             [fig_coord1[1], fig_coord2[1]], 
                             color=line_color, linestyle=line_style, linewidth=2,
                             transform=fig.transFigure))
    
    # Add results text
    result_text = f"Similarity: {similarity:.4f}\nConfidence: {confidence:.2f}%\n"
    result_text += "MATCH" if is_match else "NO MATCH"
    
    plt.figtext(0.5, 0.02, result_text, ha='center', va='bottom', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='lightgreen' if is_match else 'lightcoral'))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

# Compare two face images
def compare_faces(img_path1, img_path2):
    img1 = load_image(img_path1)
    img2 = load_image(img_path2)

    print("Extracting embeddings...")
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)

    similarity = cosine_similarity(emb1, emb2)

    #determine confidence for current comparison
    confidence = round(((similarity + 1) / 2) * 100, 2)

    print(f"\nCosine Similarity: {similarity:.4f}")
    print(f"Confidence Score: {confidence:.2f}%")

    is_match = similarity >= THRESHOLD
    if is_match:
        print("MATCH: High confidence the same person.")
    else:
        print("NO MATCH: Likely different people.")
    
    # Show visual comparison
    visualize_comparison(img_path1, img_path2, similarity, confidence, is_match)
    
    return similarity, confidence, is_match

if __name__ == "__main__":
    path_to_image1 = "face1.jpg"
    path_to_image2 = "face3.jpg"

    compare_faces(path_to_image1, path_to_image2)
