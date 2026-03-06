import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
DATA_PATH = r'D:\masters\ssat-face\data\embeddings_adaptive_00-00-000.pkl'

# EasyCom usually has ~5-6 participants per session. 
# We set this to calculate the PCR (Predicted Cluster Ratio).
GROUND_TRUTH_COUNT = 5

# EMA smoothing factor for centroid updates (0.9 = heavy weight on history)
EMA_BETA = 0.9

# Relaxation factor for adaptive threshold
RELAXATION_FACTOR = 0.95
# ---------------------


def l2_normalize(vec):
    """L2-normalize a vector."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_track_stats(embeddings):
    """
    Compute the mean embedding and internal consistency (threshold) for a track.
    
    Returns:
        mean_embedding: L2-normalized mean of all face embeddings in the track
        internal_threshold: Average pairwise cosine similarity within the track
    """
    feats = np.array(embeddings)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    
    mean_embedding = l2_normalize(np.mean(feats, axis=0))
    
    if len(feats) > 1:
        sims = cosine_similarity(feats)
        indices = np.triu_indices(len(feats), k=1)
        internal_threshold = np.mean(sims[indices])
    else:
        internal_threshold = 0.8
    
    return mean_embedding, internal_threshold


def get_track_first_frame(track_faces):
    """Get the earliest frame number for a track (for chronological sorting)."""
    return min(f['frame'] for f in track_faces)


def main():
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        faces = pickle.load(f)
    
    # ==========================================================================
    # STEP 1: Group Faces by Track ID and Collect Frame Information
    # ==========================================================================
    tracks = {}
    track_faces_raw = {}
    
    for face in faces:
        tid = face['track_id']
        if tid not in tracks:
            tracks[tid] = []
            track_faces_raw[tid] = []
        tracks[tid].append(face['embedding'])
        track_faces_raw[tid].append(face)

    print(f"Found {len(tracks)} unique tracks.")
    
    # Filter short tracks (Noise removal)
    valid_track_ids = [t for t in tracks if len(tracks[t]) > 5]
    print(f"Kept {len(valid_track_ids)} valid tracks (>5 frames).")

    # ==========================================================================
    # STEP 2: Sort Tracks Chronologically (Simulate Live Stream)
    # ==========================================================================
    track_first_frames = {tid: get_track_first_frame(track_faces_raw[tid]) 
                          for tid in valid_track_ids}
    
    sorted_track_ids = sorted(valid_track_ids, key=lambda t: track_first_frames[t])
    
    print(f"\n{'='*60}")
    print(" ONLINE STREAMING CLUSTERING (Leader-Follower + EMA)")
    print(f"{'='*60}")
    print(f"Processing {len(sorted_track_ids)} tracks in chronological order...")
    print(f"EMA Beta: {EMA_BETA} | Relaxation Factor: {RELAXATION_FACTOR}")
    print(f"{'='*60}\n")

    # ==========================================================================
    # STEP 3: Initialize Active Clusters Memory
    # ==========================================================================
    # Each cluster stores:
    #   - centroid: EMA-updated L2-normalized embedding
    #   - internal_threshold: Running consistency score
    #   - track_list: List of track IDs assigned to this cluster
    #   - n_tracks: Number of tracks (for weighted threshold updates)
    
    active_clusters = []

    # ==========================================================================
    # STEP 4: Online Leader-Follower Clustering Loop
    # ==========================================================================
    for i, tid in enumerate(sorted_track_ids):
        track_embedding, track_threshold = compute_track_stats(tracks[tid])
        
        if len(active_clusters) == 0:
            # First track -> initialize first cluster
            active_clusters.append({
                'centroid': track_embedding,
                'internal_threshold': track_threshold,
                'track_list': [tid],
                'n_tracks': 1
            })
            print(f"[Track {i+1:3d}/{len(sorted_track_ids)}] ID={tid:4d} | "
                  f"NEW CLUSTER #1 (first track)")
            continue
        
        # Compute cosine similarity to all active cluster centroids
        cluster_centroids = np.array([c['centroid'] for c in active_clusters])
        similarities = cosine_similarity(track_embedding.reshape(1, -1), 
                                         cluster_centroids)[0]
        
        # Find best matching cluster
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        best_cluster = active_clusters[best_idx]
        
        # Adaptive Decision Boundary
        adaptive_limit = min(track_threshold, 
                            best_cluster['internal_threshold']) * RELAXATION_FACTOR
        
        if best_sim > adaptive_limit:
            # ASSIGN to existing cluster
            old_centroid = best_cluster['centroid']
            
            # EMA Centroid Update: new = beta * old + (1 - beta) * incoming
            new_centroid = EMA_BETA * old_centroid + (1 - EMA_BETA) * track_embedding
            new_centroid = l2_normalize(new_centroid)
            
            # Update cluster state
            best_cluster['centroid'] = new_centroid
            best_cluster['track_list'].append(tid)
            best_cluster['n_tracks'] += 1
            
            # Update internal threshold (weighted average with new track)
            n = best_cluster['n_tracks']
            best_cluster['internal_threshold'] = (
                (n - 1) / n * best_cluster['internal_threshold'] + 
                1 / n * track_threshold
            )
            
            print(f"[Track {i+1:3d}/{len(sorted_track_ids)}] ID={tid:4d} | "
                  f"ASSIGNED to Cluster #{best_idx+1:2d} "
                  f"(sim={best_sim:.3f} > limit={adaptive_limit:.3f})")
        else:
            # CREATE new cluster
            new_cluster_id = len(active_clusters) + 1
            active_clusters.append({
                'centroid': track_embedding,
                'internal_threshold': track_threshold,
                'track_list': [tid],
                'n_tracks': 1
            })
            print(f"[Track {i+1:3d}/{len(sorted_track_ids)}] ID={tid:4d} | "
                  f"NEW CLUSTER #{new_cluster_id:2d} "
                  f"(sim={best_sim:.3f} <= limit={adaptive_limit:.3f})")

    # ==========================================================================
    # STEP 5: Format Results for Paper
    # ==========================================================================
    num_predicted = len(active_clusters)
    pcr = num_predicted / GROUND_TRUTH_COUNT
    
    # WCP (Weighted Cluster Purity) - requires ground truth labels
    wcp_display = "N/A"

    print("\n" + "="*60)
    print(" PAPER RESULTS TABLE FORMAT")
    print("="*60)
    print(f"{'Method':<25} | {'Metric':<30}")
    print("-" * 60)
    print(f"{'Online Leader-Follower':<25} | {wcp_display} & {pcr:.2f}")
    print("-" * 60)
    print(f"Total Tracks Processed:   {len(sorted_track_ids)}")
    print(f"Final Clusters Found:     {num_predicted}")
    print(f"Target (GT) Count:        {GROUND_TRUTH_COUNT}")
    print(f"Predicted Cluster Ratio:  {pcr:.2f}")
    print("="*60)
    
    # Cluster size distribution
    print("\n" + "="*60)
    print(" CLUSTER SIZE DISTRIBUTION")
    print("="*60)
    cluster_sizes = sorted([c['n_tracks'] for c in active_clusters], reverse=True)
    for idx, size in enumerate(cluster_sizes):
        bar = "█" * min(size, 50)
        print(f"Cluster {idx+1:2d}: {size:3d} tracks | {bar}")
    print("="*60)


if __name__ == "__main__":
    main()