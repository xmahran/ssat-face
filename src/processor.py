import cv2
import pickle
import insightface
from insightface.app import FaceAnalysis
import os
import numpy as np

VIDEO_PATH = r'D:\masters\ssat-face\data\EasyComDataset\Videos\01-22-15.mp4'  # still need to change

OUTPUT_PATH = r'D:\masters\ssat-face\data\embeddings_01-22-15.pkl'
# --------------------------------

def main():
    print("Initializing ArcFace...")
    # Initialize the "Modern" feature extractor
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Could not find video at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {VIDEO_PATH} ({total_frames} frames)...")

    all_faces = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Detect faces
        faces = app.get(frame)

        for face in faces:
            # Save only the essentials: Frame ID, BBox, and the 512-D Embedding
            face_data = {
                "frame": frame_idx,
                "bbox": face.bbox,
                "embedding": face.embedding, 
                "score": face.det_score
            }
            all_faces.append(face_data)

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames... Found {len(all_faces)} faces so far.")
        frame_idx += 1

    cap.release()
    
    # Save the "Face Fingerprints" to a file
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(all_faces, f)
    print(f"Done! Saved {len(all_faces)} faces to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()