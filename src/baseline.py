import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering

DATA_PATH = r'D:\masters\ssat-face\results\embeddings_00-00-000.pkl'

# The "Fixed Threshold" (Standard value is 0.6)
THRESHOLD = 0.6
# -----------------------------

def main():
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        faces = pickle.load(f)
    
    # Extract just the embeddings (the numbers)
    embeddings = np.array([f['embedding'] for f in faces])
    print(f"Loaded {len(embeddings)} faces.")

    # --- CRITICAL STEP: Normalization ---
    # ArcFace compares angles, so we must normalize vectors to length 1
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    # Run HAC Clustering (The "Modern Standard")
    print(f"Clustering with Fixed Threshold = {THRESHOLD}...")
    clusterer = AgglomerativeClustering(
        n_clusters=None,           # "None" means: figure it out automatically...
        distance_threshold=THRESHOLD, # ...based on this threshold
        metric='euclidean',        # Euclidean on normalized vectors == Cosine Distance
        linkage='average'          # Standard linkage for faces
    )
    labels = clusterer.fit_predict(embeddings)

    # Show Results
    n_unique_people = len(set(labels))
    print("\n---------------- RESULTS ----------------")
    print(f"Video File: {DATA_PATH}")
    print(f"Total Faces Found: {len(faces)}")
    print(f"Estimated Number of People: {n_unique_people}")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()