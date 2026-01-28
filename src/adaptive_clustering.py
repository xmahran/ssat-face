import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
DATA_PATH = r'D:\masters\ssat-face\data\embeddings_adaptive_00-00-000.pkl'

# EasyCom usually has ~5-6 participants per session. 
# We set this to calculate the PCR (Predicted Cluster Ratio).
GROUND_TRUTH_COUNT = 5 
# ---------------------

def main():
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        faces = pickle.load(f)
    
    # 1. Group Faces by Track ID
    tracks = {}
    for f in faces:
        tid = f['track_id']
        if tid not in tracks: tracks[tid] = []
        tracks[tid].append(f['embedding'])

    print(f"Found {len(tracks)} unique tracks.")
    
    # Filter short tracks (Noise removal)
    track_ids = [t for t in tracks if len(tracks[t]) > 5]
    print(f"Kept {len(track_ids)} valid tracks (>5 frames).")

    # 2. Compute Adaptive Thresholds (Internal Consistency)
    track_thresholds = {}
    track_embeddings = {} 

    print("Calculating Adaptive Thresholds...")
    for tid in track_ids:
        feats = np.array(tracks[tid])
        norm = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / norm
        
        # Mean embedding for the track
        mean_feat = np.mean(feats, axis=0)
        track_embeddings[tid] = mean_feat / np.linalg.norm(mean_feat)

        # Internal Consistency
        if len(feats) > 1:
            sims = cosine_similarity(feats)
            indices = np.triu_indices(len(feats), k=1)
            internal_sim = np.mean(sims[indices])
        else:
            internal_sim = 0.8
        
        track_thresholds[tid] = internal_sim

    # 3. Adaptive Clustering Loop
    clusters = [[tid] for tid in track_ids]
    
    print(f"Starting Adaptive Clustering on {len(clusters)} initial tracks...")
    
    while True:
        # Calculate similarity matrix between current clusters
        current_cluster_feats = []
        for clust in clusters:
            vecs = [track_embeddings[tid] for tid in clust]
            mean_c = np.mean(vecs, axis=0)
            current_cluster_feats.append(mean_c / np.linalg.norm(mean_c))
        
        current_cluster_feats = np.array(current_cluster_feats)
        if len(current_cluster_feats) < 2: break 

        sim_matrix = cosine_similarity(current_cluster_feats)
        np.fill_diagonal(sim_matrix, -1) 

        # Find best pair
        max_idx = np.unravel_index(np.argmax(sim_matrix, axis=None), sim_matrix.shape)
        idx_a, idx_b = max_idx
        best_sim = sim_matrix[idx_a, idx_b]
        
        # Adaptive Logic: Compare against internal consistency
        thresh_a = min([track_thresholds[tid] for tid in clusters[idx_a]])
        thresh_b = min([track_thresholds[tid] for tid in clusters[idx_b]])
        adaptive_limit = min(thresh_a, thresh_b) * 0.95 # Relaxation factor
        
        if best_sim > adaptive_limit:
            new_cluster = clusters[idx_a] + clusters[idx_b]
            # Remove carefully (largest index first)
            if idx_a > idx_b:
                clusters.pop(idx_a)
                clusters.pop(idx_b)
            else:
                clusters.pop(idx_b)
                clusters.pop(idx_a)
            clusters.append(new_cluster)
        else:
            break

    # --- 4. FORMAT RESULTS FOR PAPER ---
    # PCR (Predicted Cluster Ratio) = Pred_Count / GT_Count
    # Ideal PCR is 1.0 (e.g. 5 predicted / 5 real)
    num_predicted = len(clusters)
    pcr = num_predicted / GROUND_TRUTH_COUNT
    
    # WCP (Weighted Cluster Purity)
    # Requires loading ground truth JSON labels (Not currently implemented)
    wcp_display = "N/A" 

    print("\n" + "="*50)
    print(" PAPER RESULTS TABLE FORMAT")
    print("="*50)
    print(f"{'Method':<10} | {'Metric':<30}")
    print("-" * 45)
    # The string format from Table 3: "WCP & PCR"
    print(f"{'Ours':<10} | {wcp_display} & {pcr:.2f}")
    print("-" * 45)
    print(f"Total Tracks Processed: {len(track_ids)}")
    print(f"Final Clusters Found:   {num_predicted}")
    print(f"Target (GT) Count:      {GROUND_TRUTH_COUNT}")
    print("="*50)

if __name__ == "__main__":
    main()