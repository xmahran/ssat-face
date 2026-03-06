import json
import pickle
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
DATA_PATH = r'D:\masters\ssat-face\data\embeddings_adaptive_00-00-000.pkl'

# EasyCom Ground Truth JSON file path
GT_JSON_PATH = r'D:\masters\ssat-face\data\EasyComDataset\Videos\Main\Face_Bounding_Boxes\Session_1\00-00-000.json'

# IoU threshold for matching detected faces to ground truth
IOU_THRESHOLD = 0.5

# EMA smoothing factor for centroid updates (0.9 = heavy weight on history)
EMA_BETA = 0.9

# Relaxation factor for adaptive threshold
RELAXATION_FACTOR = 0.95
# ---------------------


def l2_normalize(vec):
    """L2-normalize a vector."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou


def load_ground_truth(gt_json_path):
    """
    Load EasyCom ground truth JSON and build a frame-indexed lookup.
    
    Returns:
        dict: {frame_number: [(participant_id, [x1, y1, x2, y2]), ...]}
    """
    print(f"Loading Ground Truth from {gt_json_path}...")
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)
    
    gt_by_frame = {}
    for frame_entry in gt_data:
        frame_num = frame_entry['Frame_Number']
        participants = []
        for p in frame_entry.get('Participants', []):
            bbox = [p['x1'], p['y1'], p['x2'], p['y2']]
            participants.append((p['Participant_ID'], bbox))
        gt_by_frame[frame_num] = participants
    
    unique_ids = set()
    for participants in gt_by_frame.values():
        for pid, _ in participants:
            unique_ids.add(pid)
    
    print(f"  Loaded {len(gt_by_frame)} frames with {len(unique_ids)} unique participants: {sorted(unique_ids)}")
    return gt_by_frame, len(unique_ids)


def assign_gt_to_faces(faces, gt_by_frame, iou_threshold=0.5):
    """
    Match each detected face to ground truth using frame + IoU matching.
    
    Args:
        faces: List of face dictionaries with 'frame' and 'bbox' keys
        gt_by_frame: Ground truth lookup from load_ground_truth()
        iou_threshold: Minimum IoU to consider a match
    
    Returns:
        faces: Same list with 'gt_id' added to each face
        stats: Dictionary with matching statistics
    """
    matched = 0
    unmatched = 0
    
    for face in faces:
        frame_num = face['frame'] + 1  # GT uses 1-indexed frames
        bbox = face['bbox']
        
        if frame_num not in gt_by_frame:
            face['gt_id'] = 'Unknown'
            unmatched += 1
            continue
        
        best_iou = 0
        best_gt_id = 'Unknown'
        
        for gt_id, gt_bbox in gt_by_frame[frame_num]:
            iou = compute_iou(bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id
        
        if best_iou >= iou_threshold:
            face['gt_id'] = best_gt_id
            matched += 1
        else:
            face['gt_id'] = 'Unknown'
            unmatched += 1
    
    stats = {
        'matched': matched,
        'unmatched': unmatched,
        'match_rate': matched / len(faces) * 100 if faces else 0
    }
    return faces, stats


def compute_track_gt_identity(track_faces):
    """
    Determine the Ground Truth identity of a track via majority voting.
    
    Args:
        track_faces: List of face dictionaries belonging to this track
    
    Returns:
        str/int: The most common gt_id in the track, or 'Unknown' if no consensus
    """
    gt_ids = [f['gt_id'] for f in track_faces if f['gt_id'] != 'Unknown']
    
    if not gt_ids:
        return 'Unknown'
    
    counter = Counter(gt_ids)
    majority_id, _ = counter.most_common(1)[0]
    return majority_id


def compute_wcp(active_clusters, track_gt_identities):
    """
    Compute Weighted Cluster Purity (WCP) as defined in VideoClusterNet.
    
    WCP = sum(purity_i * n_i) / N
    
    Where:
        - purity_i = (count of dominant GT in cluster i) / (total tracks in cluster i)
        - n_i = number of tracks in cluster i
        - N = total number of tracks
    
    Args:
        active_clusters: List of cluster dictionaries with 'track_list'
        track_gt_identities: Dict mapping track_id -> gt_identity
    
    Returns:
        float: WCP score (0.0 to 1.0)
        dict: Detailed breakdown per cluster
    """
    total_tracks = sum(c['n_tracks'] for c in active_clusters)
    
    if total_tracks == 0:
        return 0.0, {}
    
    weighted_purity_sum = 0.0
    cluster_details = []
    
    for idx, cluster in enumerate(active_clusters):
        track_ids = cluster['track_list']
        n_tracks = len(track_ids)
        
        gt_ids_in_cluster = [track_gt_identities.get(tid, 'Unknown') for tid in track_ids]
        gt_ids_valid = [gid for gid in gt_ids_in_cluster if gid != 'Unknown']
        
        if not gt_ids_valid:
            purity = 0.0
            dominant_id = 'Unknown'
            dominant_count = 0
        else:
            counter = Counter(gt_ids_valid)
            dominant_id, dominant_count = counter.most_common(1)[0]
            purity = dominant_count / n_tracks
        
        weighted_purity_sum += purity * n_tracks
        
        cluster_details.append({
            'cluster_idx': idx + 1,
            'n_tracks': n_tracks,
            'dominant_gt': dominant_id,
            'dominant_count': dominant_count,
            'purity': purity,
            'all_gt_ids': dict(Counter(gt_ids_in_cluster))
        })
    
    wcp = weighted_purity_sum / total_tracks
    return wcp, cluster_details


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
    # STEP 1: Load Ground Truth and Match Faces
    # ==========================================================================
    gt_by_frame, ground_truth_count = load_ground_truth(GT_JSON_PATH)
    faces, match_stats = assign_gt_to_faces(faces, gt_by_frame, IOU_THRESHOLD)
    
    print(f"\n  GT Matching Results:")
    print(f"    Matched:   {match_stats['matched']} faces")
    print(f"    Unmatched: {match_stats['unmatched']} faces")
    print(f"    Match Rate: {match_stats['match_rate']:.1f}%")

    # ==========================================================================
    # STEP 2: Group Faces by Track ID and Collect Frame Information
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

    print(f"\nFound {len(tracks)} unique tracks.")
    
    # Filter short tracks (Noise removal)
    valid_track_ids = [t for t in tracks if len(tracks[t]) > 5]
    print(f"Kept {len(valid_track_ids)} valid tracks (>5 frames).")

    # ==========================================================================
    # STEP 3: Compute Track-Level Ground Truth Identities (Majority Vote)
    # ==========================================================================
    track_gt_identities = {}
    for tid in valid_track_ids:
        track_gt_identities[tid] = compute_track_gt_identity(track_faces_raw[tid])
    
    gt_distribution = Counter(track_gt_identities.values())
    print(f"\nTrack GT Distribution (majority vote):")
    for gt_id, count in sorted(gt_distribution.items(), key=lambda x: -x[1]):
        print(f"  Participant {gt_id}: {count} tracks")

    # ==========================================================================
    # STEP 4: Sort Tracks Chronologically (Simulate Live Stream)
    # ==========================================================================
    track_first_frames = {tid: get_track_first_frame(track_faces_raw[tid]) 
                          for tid in valid_track_ids}
    
    sorted_track_ids = sorted(valid_track_ids, key=lambda t: track_first_frames[t])
    
    print(f"\n{'='*70}")
    print(" ONLINE STREAMING CLUSTERING (Leader-Follower + EMA)")
    print(f"{'='*70}")
    print(f"Processing {len(sorted_track_ids)} tracks in chronological order...")
    print(f"EMA Beta: {EMA_BETA} | Relaxation Factor: {RELAXATION_FACTOR} | IoU Thresh: {IOU_THRESHOLD}")
    print(f"{'='*70}\n")

    # ==========================================================================
    # STEP 5: Initialize Active Clusters Memory
    # ==========================================================================
    active_clusters = []

    # ==========================================================================
    # STEP 6: Online Leader-Follower Clustering Loop
    # ==========================================================================
    for i, tid in enumerate(sorted_track_ids):
        track_embedding, track_threshold = compute_track_stats(tracks[tid])
        gt_label = track_gt_identities[tid]
        
        if len(active_clusters) == 0:
            active_clusters.append({
                'centroid': track_embedding,
                'internal_threshold': track_threshold,
                'track_list': [tid],
                'n_tracks': 1
            })
            print(f"[Track {i+1:3d}/{len(sorted_track_ids)}] ID={tid:4d} (GT:{gt_label}) | "
                  f"NEW CLUSTER #1 (first track)")
            continue
        
        cluster_centroids = np.array([c['centroid'] for c in active_clusters])
        similarities = cosine_similarity(track_embedding.reshape(1, -1), 
                                         cluster_centroids)[0]
        
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        best_cluster = active_clusters[best_idx]
        
        adaptive_limit = min(track_threshold, 
                            best_cluster['internal_threshold']) * RELAXATION_FACTOR
        
        if best_sim > adaptive_limit:
            old_centroid = best_cluster['centroid']
            new_centroid = EMA_BETA * old_centroid + (1 - EMA_BETA) * track_embedding
            new_centroid = l2_normalize(new_centroid)
            
            best_cluster['centroid'] = new_centroid
            best_cluster['track_list'].append(tid)
            best_cluster['n_tracks'] += 1
            
            n = best_cluster['n_tracks']
            best_cluster['internal_threshold'] = (
                (n - 1) / n * best_cluster['internal_threshold'] + 
                1 / n * track_threshold
            )
            
            print(f"[Track {i+1:3d}/{len(sorted_track_ids)}] ID={tid:4d} (GT:{gt_label}) | "
                  f"ASSIGNED to Cluster #{best_idx+1:2d} "
                  f"(sim={best_sim:.3f} > limit={adaptive_limit:.3f})")
        else:
            new_cluster_id = len(active_clusters) + 1
            active_clusters.append({
                'centroid': track_embedding,
                'internal_threshold': track_threshold,
                'track_list': [tid],
                'n_tracks': 1
            })
            print(f"[Track {i+1:3d}/{len(sorted_track_ids)}] ID={tid:4d} (GT:{gt_label}) | "
                  f"NEW CLUSTER #{new_cluster_id:2d} "
                  f"(sim={best_sim:.3f} <= limit={adaptive_limit:.3f})")

    # ==========================================================================
    # STEP 7: Calculate Weighted Cluster Purity (WCP)
    # ==========================================================================
    wcp, cluster_details = compute_wcp(active_clusters, track_gt_identities)

    # ==========================================================================
    # STEP 8: Format Results for Paper
    # ==========================================================================
    num_predicted = len(active_clusters)
    pcr = num_predicted / ground_truth_count

    print("\n" + "="*70)
    print(" PAPER RESULTS TABLE FORMAT (VideoClusterNet Style)")
    print("="*70)
    print(f"{'Method':<30} | {'WCP':<12} | {'PCR':<12}")
    print("-" * 70)
    print(f"{'Online Leader-Follower + EMA':<30} | {wcp*100:>10.2f}% | {pcr:>10.2f}")
    print("-" * 70)
    print(f"Total Tracks Processed:    {len(sorted_track_ids)}")
    print(f"Final Clusters Found:      {num_predicted}")
    print(f"Ground Truth Participants: {ground_truth_count}")
    print(f"Predicted Cluster Ratio:   {pcr:.2f}")
    print(f"Weighted Cluster Purity:   {wcp*100:.2f}%")
    print("="*70)
    
    # Detailed Cluster Breakdown
    print("\n" + "="*70)
    print(" CLUSTER PURITY BREAKDOWN")
    print("="*70)
    print(f"{'Cluster':<10} | {'Tracks':<8} | {'Dominant GT':<12} | {'Purity':<10} | {'Composition'}")
    print("-" * 70)
    
    for detail in sorted(cluster_details, key=lambda x: -x['n_tracks']):
        composition = ", ".join([f"P{k}:{v}" for k, v in detail['all_gt_ids'].items()])
        print(f"#{detail['cluster_idx']:<9} | {detail['n_tracks']:<8} | "
              f"P{detail['dominant_gt']:<11} | {detail['purity']*100:>8.1f}% | {composition}")
    
    print("="*70)
    
    # Cluster size distribution (visual)
    print("\n" + "="*70)
    print(" CLUSTER SIZE DISTRIBUTION")
    print("="*70)
    for detail in sorted(cluster_details, key=lambda x: -x['n_tracks']):
        bar = "█" * min(detail['n_tracks'], 50)
        purity_indicator = "✓" if detail['purity'] >= 0.9 else "○" if detail['purity'] >= 0.7 else "✗"
        print(f"Cluster #{detail['cluster_idx']:2d} [P{detail['dominant_gt']}] {purity_indicator}: "
              f"{detail['n_tracks']:3d} tracks | {bar}")
    print("="*70)
    print(f"\nLegend: ✓ = Purity ≥90%, ○ = Purity ≥70%, ✗ = Purity <70%")


if __name__ == "__main__":
    main()