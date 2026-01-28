import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis
import os

VIDEO_PATH = r'D:\masters\ssat-face\data\EasyComDataset\Videos\mp4\Main\Video_Compressed\Session_1\00-00-000.mp4'  

OUTPUT_PATH = r'D:\masters\ssat-face\data\embeddings_adaptive_00-00-000.pkl'
# --------------------------------

# --- SIMPLE TRACKER CLASS (Corrected) ---
class SimpleTracker:
    def __init__(self, iou_thresh=0.3):
        self.active_tracks = [] # List of {id, bbox, last_frame}
        self.track_counter = 0
        self.iou_thresh = iou_thresh

    def update(self, faces, frame_idx):
        # Calculate IoU between all new faces and active tracks
        updated_faces = []
        
        for face in faces:
            best_iou = 0
            best_track_id = -1
            
            # 1. Try to match with an existing track
            for t in self.active_tracks:
                # Only look at tracks seen recently (within last 5 frames)
                if frame_idx - t['last_frame'] > 5: continue
                
                # Calculate Intersection over Union (IoU)
                # FIXED: Use ['bbox'] because 'face' is now a dictionary
                boxA = face['bbox'] 
                boxB = t['bbox']
                
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                
                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
                boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
                
                iou = interArea / float(boxAArea + boxBArea - interArea)
                
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = t['id']

            # 2. Assign Track ID
            if best_iou > self.iou_thresh:
                face['track_id'] = best_track_id
                # Update the track's last known position
                for t in self.active_tracks:
                    if t['id'] == best_track_id:
                        t['bbox'] = face['bbox'] # FIXED
                        t['last_frame'] = frame_idx
            else:
                # Create new track
                self.track_counter += 1
                face['track_id'] = self.track_counter
                self.active_tracks.append({
                    'id': self.track_counter, 
                    'bbox': face['bbox'], # FIXED
                    'last_frame': frame_idx
                })
            
            updated_faces.append(face)
        return updated_faces

def main():
    print("Initializing ArcFace & Tracker...")
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    tracker = SimpleTracker()

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {VIDEO_PATH} ({total_frames} frames)...")

    all_faces = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        faces = app.get(frame)
        
        # --- Convert to dictionary to add track_id ---
        face_dicts = []
        for face in faces:
            # InsightFace returns an object, we convert it to dict here
            face_dicts.append({
                "bbox": face.bbox,
                "embedding": face.embedding,
                "score": face.det_score,
                "frame": frame_idx
            })
            
        # --- Run Tracker ---
        tracked_faces = tracker.update(face_dicts, frame_idx)
        all_faces.extend(tracked_faces)

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames... Found {len(all_faces)} faces.")
        frame_idx += 1

    cap.release()
    
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(all_faces, f)
    print(f"Done! Saved {len(all_faces)} tracked faces to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()