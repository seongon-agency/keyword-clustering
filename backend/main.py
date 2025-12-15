"""
FastAPI Backend for Keyword Clustering
Uses the same Python dependencies as the Streamlit app

Performance optimizations:
- Parallel embedding requests (concurrent API calls)
- GPT-4o-mini for faster labeling
- Parallelized UMAP

Features:
- Real-time progress streaming via SSE
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Generator
import numpy as np
import os
import json
import time
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Load environment variables from parent directory (.env is in root)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Fallback: try current working directory
    load_dotenv()

# NLP and Clustering libraries
import hdbscan
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from umap import UMAP

# Vietnamese NLP (optional)
try:
    from underthesea import word_tokenize
    VIETNAMESE_AVAILABLE = True
except ImportError:
    VIETNAMESE_AVAILABLE = False

app = FastAPI(
    title="Keyword Clustering API",
    description="AI-Powered Keyword Clustering with OpenAI & GPT-4o-mini",
    version="1.2.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3002", "http://127.0.0.1:3000", "http://127.0.0.1:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClusteringConfig(BaseModel):
    """
    User-friendly clustering configuration with high-impact parameters.
    """
    # Target number of clusters (0 = auto-detect)
    # This is the most intuitive parameter for users
    target_clusters: int = 0  # 0 means auto

    # Granularity: How fine-grained should clusters be? (1=few large, 10=many small)
    # This has the biggest visual impact on results
    granularity: int = 5

    # Minimum keywords required to form a cluster
    min_keywords_per_cluster: int = 5

    # Cluster coherence: How similar must keywords be? (1=loose, 10=strict)
    cluster_coherence: int = 5


# Preset configurations - make them VERY different for visible impact
CLUSTERING_PRESETS = {
    "recommended": ClusteringConfig(
        target_clusters=0,
        granularity=5,
        min_keywords_per_cluster=8,
        cluster_coherence=5
    ),
    "few_large": ClusteringConfig(
        target_clusters=0,
        granularity=2,
        min_keywords_per_cluster=20,
        cluster_coherence=3
    ),
    "many_small": ClusteringConfig(
        target_clusters=0,
        granularity=9,
        min_keywords_per_cluster=5,
        cluster_coherence=7
    ),
    "strict_quality": ClusteringConfig(
        target_clusters=0,
        granularity=6,
        min_keywords_per_cluster=10,
        cluster_coherence=9
    )
}


class ClusterRequest(BaseModel):
    keywords: List[str]
    api_key: Optional[str] = None
    language: str = "Vietnamese"
    clustering_blocks: int = 1000
    clustering_config: Optional[ClusteringConfig] = None  # If None, use recommended


class ClusterResponse(BaseModel):
    keywords: List[str]
    segmented: List[str]
    clusters: List[int]
    cluster_labels: Dict[int, str]
    embeddings_2d: List[List[float]]
    embeddings_3d: List[List[float]]


def get_api_key(request_key: Optional[str] = None) -> str:
    """Get OpenAI API key from request or environment"""
    if request_key and request_key.strip():
        return request_key.strip()
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY in .env or provide api_key in request.")


def apply_vietnamese_word_segmentation(keywords: List[str], language: str) -> List[str]:
    """Apply Vietnamese word segmentation using underthesea"""
    if language != "Vietnamese" or not VIETNAMESE_AVAILABLE:
        return keywords

    segmented = []
    for keyword in keywords:
        try:
            tokens = word_tokenize(keyword)
            segmented_keyword = " ".join(tokens)
            segmented.append(segmented_keyword)
        except:
            segmented.append(keyword)

    return segmented


def get_openai_embeddings_parallel(
    keywords: List[str],
    api_key: str,
    model: str = "text-embedding-3-large",
    progress_callback=None
) -> np.ndarray:
    """Get embeddings from OpenAI API with parallel batch requests"""
    client = OpenAI(api_key=api_key)
    batch_size = 100
    max_workers = 5

    batches = []
    for i in range(0, len(keywords), batch_size):
        batches.append((i, keywords[i:i + batch_size]))

    results = {}
    completed = 0
    total_batches = len(batches)

    def fetch_batch(batch_info):
        batch_idx, batch = batch_info
        response = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float"
        )
        return batch_idx, [item.embedding for item in response.data]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_batch, batch): batch[0] for batch in batches}

        for future in as_completed(futures):
            batch_idx, embeddings = future.result()
            results[batch_idx] = embeddings
            completed += 1
            if progress_callback:
                progress_callback(completed, total_batches)

    all_embeddings = []
    for i in sorted(results.keys()):
        all_embeddings.extend(results[i])

    return np.array(all_embeddings)


def reassign_outliers_to_nearest_cluster(embeddings: np.ndarray, clusters: np.ndarray, k: int = 5, min_similarity: float = 0.3) -> np.ndarray:
    """
    Reassign outlier points to their nearest cluster using k-NN voting.
    Only reassigns if the outlier is sufficiently similar to the target cluster.
    """
    outlier_indices = np.where(clusters == -1)[0]
    non_outlier_indices = np.where(clusters != -1)[0]

    if len(outlier_indices) == 0 or len(non_outlier_indices) == 0:
        return clusters

    outlier_embeddings = embeddings[outlier_indices]
    non_outlier_embeddings = embeddings[non_outlier_indices]
    similarities = cosine_similarity(outlier_embeddings, non_outlier_embeddings)

    # Calculate cluster centroids for similarity check
    unique_clusters = [c for c in np.unique(clusters) if c != -1]
    centroids = {}
    for c in unique_clusters:
        mask = clusters == c
        centroids[c] = embeddings[mask].mean(axis=0)

    new_clusters = clusters.copy()
    k_actual = min(k, len(non_outlier_indices))
    kept_as_outlier = 0

    for i, outlier_idx in enumerate(outlier_indices):
        # Get top-k nearest neighbors
        top_k_indices = np.argsort(similarities[i])[-k_actual:]
        top_k_clusters = [clusters[non_outlier_indices[idx]] for idx in top_k_indices]

        # Majority vote among k nearest neighbors
        cluster_votes = Counter(top_k_clusters)
        best_cluster = cluster_votes.most_common(1)[0][0]

        # Quality check: only reassign if similar enough to cluster centroid
        outlier_vec = outlier_embeddings[i].reshape(1, -1)
        centroid_vec = centroids[best_cluster].reshape(1, -1)
        sim_to_centroid = cosine_similarity(outlier_vec, centroid_vec)[0][0]

        if sim_to_centroid >= min_similarity:
            new_clusters[outlier_idx] = best_cluster
        else:
            # Keep as outlier - will be assigned to largest cluster later
            kept_as_outlier += 1

    if kept_as_outlier > 0:
        print(f"[DEBUG] {kept_as_outlier} outliers kept (low similarity to nearest cluster)")

    return new_clusters


def merge_similar_clusters(embeddings: np.ndarray, clusters: np.ndarray, similarity_threshold: float = 0.85) -> np.ndarray:
    """
    Merge clusters whose centroids are too similar.
    This reduces the number of clusters by combining semantically similar groups.
    """
    unique_clusters = [c for c in np.unique(clusters) if c >= 0]
    if len(unique_clusters) <= 1:
        return clusters

    # Calculate cluster centroids
    centroids = {}
    for c in unique_clusters:
        mask = clusters == c
        centroids[c] = embeddings[mask].mean(axis=0)

    # Build similarity matrix and merge similar clusters
    new_clusters = clusters.copy()
    merged = set()

    for i, c1 in enumerate(unique_clusters):
        if c1 in merged:
            continue
        for c2 in unique_clusters[i+1:]:
            if c2 in merged:
                continue
            # Cosine similarity between centroids
            sim = np.dot(centroids[c1], centroids[c2]) / (
                np.linalg.norm(centroids[c1]) * np.linalg.norm(centroids[c2]) + 1e-10
            )
            if sim >= similarity_threshold:
                # Merge c2 into c1
                new_clusters[new_clusters == c2] = c1
                merged.add(c2)

    # Renumber clusters to be consecutive
    unique_new = sorted([c for c in np.unique(new_clusters) if c >= 0])
    mapping = {old: new for new, old in enumerate(unique_new)}
    mapping[-1] = -1
    return np.array([mapping[c] for c in new_clusters])


def force_bisect(embeddings: np.ndarray, max_cluster_size: int, min_cluster_size: int = 10) -> np.ndarray:
    """
    Recursively bisect clusters using K-Means (k=2) until all are under max_cluster_size.
    This guarantees balanced splits unlike regular K-Means with higher k.
    """
    n_samples = len(embeddings)
    labels = np.zeros(n_samples, dtype=int)

    print(f"[DEBUG] force_bisect called: n_samples={n_samples}, max_cluster_size={max_cluster_size}")

    if n_samples <= max_cluster_size:
        print(f"[DEBUG] force_bisect: Already under max size, returning all zeros")
        return labels

    # Queue of (indices, cluster_id) to process
    next_cluster_id = 0
    queue = [(np.arange(n_samples), next_cluster_id)]
    next_cluster_id += 1
    bisect_count = 0

    while queue:
        indices, current_id = queue.pop(0)

        if len(indices) <= max_cluster_size:
            labels[indices] = current_id
            continue

        bisect_count += 1
        # Bisect this cluster
        cluster_embeddings = embeddings[indices]

        # Quick dimensionality reduction for bisection
        if cluster_embeddings.shape[1] > 10:
            n_comp = min(10, len(indices) - 1)
            try:
                reducer = UMAP(n_components=n_comp, n_neighbors=min(5, len(indices)-1),
                              min_dist=0.0, metric='euclidean', random_state=42, n_jobs=1)
                reduced = reducer.fit_transform(cluster_embeddings)
            except Exception as e:
                print(f"[DEBUG] UMAP failed in bisect: {e}, using raw embeddings")
                reduced = cluster_embeddings
        else:
            reduced = cluster_embeddings

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        bisect_labels = kmeans.fit_predict(reduced)

        # Split into two groups
        group0_indices = indices[bisect_labels == 0]
        group1_indices = indices[bisect_labels == 1]

        print(f"[DEBUG] Bisect #{bisect_count}: {len(indices)} -> [{len(group0_indices)}, {len(group1_indices)}]")

        # Check if split is too imbalanced (one side < min_cluster_size)
        if len(group0_indices) < min_cluster_size or len(group1_indices) < min_cluster_size:
            print(f"[DEBUG] Bisect failed (too imbalanced), keeping as cluster {current_id}")
            labels[indices] = current_id
            continue

        # Add both groups to queue for further processing
        queue.append((group0_indices, current_id))
        queue.append((group1_indices, next_cluster_id))
        next_cluster_id += 1

    # Final stats
    unique_labels = np.unique(labels)
    final_sizes = {lbl: np.sum(labels == lbl) for lbl in unique_labels}
    print(f"[DEBUG] force_bisect done: {len(unique_labels)} clusters, sizes={sorted(final_sizes.values(), reverse=True)}")

    return labels


def subdivide_cluster(embeddings: np.ndarray, min_cluster_size: int = 10, max_cluster_size: int = 300) -> np.ndarray:
    """
    Subdivide a cluster into smaller sub-clusters using recursive bisection.
    Guarantees all resulting sub-clusters are under max_cluster_size.
    """
    n_samples = len(embeddings)
    if n_samples < min_cluster_size * 2:
        print(f"[DEBUG] Cluster too small to subdivide: {n_samples} < {min_cluster_size * 2}")
        return np.zeros(n_samples, dtype=int)

    if n_samples <= max_cluster_size:
        print(f"[DEBUG] Cluster already under max size: {n_samples} <= {max_cluster_size}")
        return np.zeros(n_samples, dtype=int)

    print(f"[DEBUG] Subdividing {n_samples} points with force_bisect (max_size={max_cluster_size})")

    # Use recursive bisection directly - guaranteed balanced results
    sub_labels = force_bisect(embeddings, max_cluster_size, min_cluster_size)

    num_sub = len(np.unique(sub_labels))
    sizes = [np.sum(sub_labels == lbl) for lbl in np.unique(sub_labels)]
    print(f"[DEBUG] force_bisect created {num_sub} sub-clusters, sizes: {sorted(sizes, reverse=True)}")
    return sub_labels


def refine_cluster_assignments(embeddings: np.ndarray, clusters: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    """
    Refine cluster assignments by checking if each point is closer to another cluster's centroid.
    Only reassigns if the point is significantly closer to another cluster.
    """
    unique_clusters = [c for c in np.unique(clusters) if c >= 0]
    if len(unique_clusters) <= 1:
        return clusters

    # Calculate cluster centroids
    centroids = {}
    for c in unique_clusters:
        mask = clusters == c
        centroids[c] = embeddings[mask].mean(axis=0)

    centroid_matrix = np.array([centroids[c] for c in unique_clusters])
    cluster_list = list(unique_clusters)

    # Check each point
    new_clusters = clusters.copy()
    reassigned = 0

    for i in range(len(embeddings)):
        current_cluster = clusters[i]
        if current_cluster == -1:
            continue

        # Calculate similarity to all centroids
        point = embeddings[i].reshape(1, -1)
        similarities = cosine_similarity(point, centroid_matrix)[0]

        current_idx = cluster_list.index(current_cluster)
        current_sim = similarities[current_idx]

        # Find best cluster
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]
        best_cluster = cluster_list[best_idx]

        # Only reassign if significantly better fit (and not already in best cluster)
        if best_cluster != current_cluster and best_sim > current_sim + 0.05:
            new_clusters[i] = best_cluster
            reassigned += 1

    if reassigned > 0:
        print(f"[DEBUG] Refinement: reassigned {reassigned} points to better-fitting clusters")

    return new_clusters


def translate_config_to_params(config: ClusteringConfig, n_samples: int) -> dict:
    """
    Translate user-friendly config to actual algorithm parameters.
    Designed to produce VISIBLY DIFFERENT results across the parameter range.
    """
    granularity = config.granularity
    coherence = config.cluster_coherence

    # GRANULARITY (1-10) - This is the most impactful parameter
    # Controls: max_cluster_size, UMAP n_neighbors, and effectively number of clusters
    # 1 = few large clusters, 10 = many small clusters

    # max_cluster_size: determines when clusters get subdivided
    # granularity 1 → 50% of dataset, granularity 10 → 5% of dataset
    max_cluster_pct = 0.50 - (granularity - 1) * 0.05  # 1→50%, 10→5%
    max_cluster_size = max(50, int(n_samples * max_cluster_pct))

    # UMAP n_neighbors: lower = more local structure = more clusters
    # granularity 1 → 30 neighbors (global), granularity 10 → 5 neighbors (local)
    umap_n_neighbors = max(5, 30 - (granularity - 1) * 3)  # 1→30, 10→3

    # UMAP min_dist: higher = more spread = easier to find cluster boundaries
    # granularity 1 → 0.0 (packed), granularity 10 → 0.5 (spread)
    umap_min_dist = (granularity - 1) * 0.055  # 1→0.0, 10→0.5

    # COHERENCE (1-10) - Controls cluster quality/tightness
    # 1 = loose grouping (more keywords assigned), 10 = strict (only very similar)

    # HDBSCAN min_samples: higher = more conservative clustering
    hdbscan_min_samples = max(1, coherence // 2 + 1)  # 1→1, 5→3, 10→6

    # Outlier similarity threshold: higher = stricter about what gets assigned
    outlier_min_similarity = 0.05 + (coherence - 1) * 0.05  # 1→0.05, 10→0.50

    # min_cluster_size from user config
    min_cluster_size = max(3, config.min_keywords_per_cluster)

    # Target clusters (if specified by user)
    target_clusters = config.target_clusters if config.target_clusters > 0 else None

    return {
        'umap_min_dist': round(umap_min_dist, 2),
        'umap_n_neighbors': umap_n_neighbors,
        'min_cluster_size': min_cluster_size,
        'max_cluster_size': max_cluster_size,
        'hdbscan_min_samples': hdbscan_min_samples,
        'outlier_min_similarity': round(outlier_min_similarity, 2),
        'target_clusters': target_clusters,
    }


def cluster_with_hdbscan(embeddings: np.ndarray, n_blocks: int, config: ClusteringConfig = None) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN with UMAP dimensionality reduction.
    Accepts user-friendly ClusteringConfig for parameter tuning.
    """
    n_samples = len(embeddings)

    # Use default config if none provided
    if config is None:
        config = CLUSTERING_PRESETS["recommended"]

    # Translate user-friendly config to algorithm parameters
    params = translate_config_to_params(config, n_samples)
    print(f"[DEBUG] Clustering config: granularity={config.granularity}, "
          f"min_per_cluster={config.min_keywords_per_cluster}, coherence={config.cluster_coherence}")
    print(f"[DEBUG] Translated params: {params}")

    # Thresholds from config
    max_cluster_size = params['max_cluster_size']
    min_cluster_size = params['min_cluster_size']
    tiny_cluster_threshold = max(3, min_cluster_size // 2)

    # UMAP with user-configured parameters
    n_components = min(30, n_samples - 1)
    umap_reducer = UMAP(
        n_components=n_components,
        n_neighbors=params['umap_n_neighbors'],
        min_dist=params['umap_min_dist'],
        metric='cosine',
        random_state=42,
        n_jobs=1
    )
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    # HDBSCAN with user-configured parameters
    hdbscan_params = {
        'min_cluster_size': min_cluster_size,
        'min_samples': params['hdbscan_min_samples'],
        'cluster_selection_epsilon': 0.0,
        'cluster_selection_method': 'eom',
        'metric': 'euclidean',
        'core_dist_n_jobs': -1
    }

    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    all_clusters = clusterer.fit_predict(reduced_embeddings)

    print(f"[DEBUG] HDBSCAN initial: {len(np.unique(all_clusters))} clusters, outliers: {np.sum(all_clusters == -1)}/{n_samples}")

    # Reassign outliers using k-NN voting with quality threshold from config
    if np.sum(all_clusters == -1) > 0:
        all_clusters = reassign_outliers_to_nearest_cluster(
            reduced_embeddings, all_clusters, k=5,
            min_similarity=params['outlier_min_similarity']
        )

    # Recursively subdivide oversized clusters
    max_iterations = 10  # Increased iterations
    for iteration in range(max_iterations):
        cluster_sizes = {}
        for c in np.unique(all_clusters):
            cluster_sizes[c] = np.sum(all_clusters == c)

        # Print current cluster size distribution
        sizes_sorted = sorted(cluster_sizes.values(), reverse=True)
        print(f"[DEBUG] Iteration {iteration}: Cluster sizes (top 5): {sizes_sorted[:5]}, max_allowed: {max_cluster_size}")

        oversized = [c for c, size in cluster_sizes.items() if size > max_cluster_size]
        if not oversized:
            print(f"[DEBUG] No oversized clusters, done subdividing")
            break

        print(f"[DEBUG] Found {len(oversized)} oversized clusters to subdivide")

        next_cluster_id = max(all_clusters) + 1
        new_clusters = all_clusters.copy()

        for cluster_id in oversized:
            cluster_mask = all_clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = reduced_embeddings[cluster_indices]

            print(f"[DEBUG] Subdividing cluster {cluster_id} with {len(cluster_indices)} points...")
            sub_labels = subdivide_cluster(cluster_embeddings, min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size)

            unique_sub = set(sub_labels)
            print(f"[DEBUG] subdivide_cluster returned {len(unique_sub)} unique labels: {unique_sub}")

            if len(unique_sub) >= 2:
                # Track sizes of each sub-cluster
                sub_sizes = {sid: np.sum(sub_labels == sid) for sid in unique_sub}
                print(f"[DEBUG] Sub-cluster sizes: {sub_sizes}")

                for sub_id in unique_sub:
                    if sub_id == 0:
                        # sub_id 0 keeps original cluster_id, but verify its size
                        size_0 = np.sum(sub_labels == 0)
                        print(f"[DEBUG] sub_id=0 has {size_0} points, keeping in cluster {cluster_id}")
                        continue
                    sub_mask = sub_labels == sub_id
                    global_indices = cluster_indices[sub_mask]
                    new_clusters[global_indices] = next_cluster_id
                    print(f"[DEBUG] sub_id={sub_id} ({len(global_indices)} pts) -> new cluster {next_cluster_id}")
                    next_cluster_id += 1

                print(f"[DEBUG] Cluster {cluster_id} ({len(cluster_indices)} pts) -> {len(unique_sub)} sub-clusters")
            else:
                print(f"[DEBUG] WARNING: subdivide_cluster returned only 1 unique label, cluster NOT split!")

        all_clusters = new_clusters

    # Merge tiny clusters into nearest larger cluster
    cluster_sizes = {c: np.sum(all_clusters == c) for c in np.unique(all_clusters)}
    tiny_clusters = [c for c, size in cluster_sizes.items() if size < tiny_cluster_threshold]

    if tiny_clusters:
        print(f"[DEBUG] Merging {len(tiny_clusters)} tiny clusters (size < {tiny_cluster_threshold})")
        # Calculate centroids for non-tiny clusters
        large_clusters = [c for c in np.unique(all_clusters) if c not in tiny_clusters]
        if large_clusters:
            centroids = {}
            for c in large_clusters:
                mask = all_clusters == c
                centroids[c] = reduced_embeddings[mask].mean(axis=0)

            for tiny_c in tiny_clusters:
                tiny_mask = all_clusters == tiny_c
                tiny_centroid = reduced_embeddings[tiny_mask].mean(axis=0)

                # Find nearest large cluster
                best_cluster = large_clusters[0]
                best_sim = -1
                for lc in large_clusters:
                    sim = np.dot(tiny_centroid, centroids[lc]) / (
                        np.linalg.norm(tiny_centroid) * np.linalg.norm(centroids[lc]) + 1e-10
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = lc

                all_clusters[tiny_mask] = best_cluster

    # Renumber clusters to be consecutive
    unique_clusters = sorted(np.unique(all_clusters))
    mapping = {old: new for new, old in enumerate(unique_clusters)}
    all_clusters = np.array([mapping[c] for c in all_clusters])

    # Final refinement pass - reassign poorly-fit points
    all_clusters = refine_cluster_assignments(reduced_embeddings, all_clusters)

    num_clusters = len(np.unique(all_clusters))
    print(f"[DEBUG] Final: {num_clusters} clusters")

    # Light merge only if too many clusters
    if num_clusters > 50:
        all_clusters = merge_similar_clusters(reduced_embeddings, all_clusters, similarity_threshold=0.93)

    return all_clusters


def generate_cluster_labels_fast(
    keywords: List[str],
    clusters: np.ndarray,
    api_key: str,
    language: str = "Vietnamese",
    progress_callback=None
) -> Dict[int, str]:
    """Generate cluster labels using GPT-4o-mini with parallel requests"""
    client = OpenAI(api_key=api_key)
    unique_clusters = np.unique(clusters)
    cluster_labels = {}
    completed = 0
    total_clusters = len(unique_clusters)

    def label_cluster(cluster_id):
        if cluster_id == -1:
            return int(cluster_id), "Outliers / Uncategorized"

        cluster_indices = np.where(clusters == cluster_id)[0]
        sample_indices = cluster_indices[:15]
        cluster_keywords = [keywords[i] for i in sample_indices]
        keywords_str = "\n".join(f"- {kw}" for kw in cluster_keywords)

        prompt = f"""Analyze these {language} keywords and create a concise, descriptive label for this semantic cluster.

Keywords in this cluster:
{keywords_str}

Requirements:
1. Provide a short label (2-5 words) in {language} that captures the main theme
2. Be specific and descriptive
3. Respond with ONLY the label, no explanation

Label:"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )

        label = response.choices[0].message.content.strip()
        return int(cluster_id), label

    max_workers = 10

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(label_cluster, cid): cid for cid in unique_clusters}

        for future in as_completed(futures):
            try:
                cluster_id, label = future.result()
                cluster_labels[cluster_id] = label
            except Exception:
                cluster_id = futures[future]
                cluster_labels[int(cluster_id)] = f"Cluster {cluster_id}"

            completed += 1
            if progress_callback:
                progress_callback(completed, total_clusters)

    return cluster_labels


def create_2d_embeddings(embeddings: np.ndarray, clusters: np.ndarray = None) -> np.ndarray:
    """Reduce embeddings to 2D using optimized UMAP"""
    umap_model = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.3,  # Increased for more spacing between points
        spread=1.5,    # Spread out clusters more
        metric='cosine',
        random_state=42,
        n_jobs=-1,
        low_memory=True,
        target_metric='categorical' if clusters is not None else None,
    )

    if clusters is not None:
        return umap_model.fit_transform(embeddings, y=clusters)
    return umap_model.fit_transform(embeddings)


def create_3d_embeddings(embeddings: np.ndarray, clusters: np.ndarray = None) -> np.ndarray:
    """Reduce embeddings to 3D using optimized UMAP for interactive visualization"""
    umap_model = UMAP(
        n_neighbors=15,
        n_components=3,
        min_dist=0.3,
        spread=1.5,
        metric='cosine',
        random_state=42,
        n_jobs=-1,
        low_memory=True,
        target_metric='categorical' if clusters is not None else None,
    )

    if clusters is not None:
        return umap_model.fit_transform(embeddings, y=clusters)
    return umap_model.fit_transform(embeddings)


def cluster_with_progress(request: ClusterRequest) -> Generator[str, None, None]:
    """Generator that yields progress updates and final results"""

    def send_event(event_type: str, data: dict):
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"

    try:
        # Get API key
        try:
            api_key = get_api_key(request.api_key)
        except ValueError as e:
            yield send_event("error", {"message": str(e)})
            return

        keywords = request.keywords
        keywords = [kw.strip() for kw in keywords if kw.strip()]
        keywords = list(dict.fromkeys(keywords))

        if len(keywords) < 2:
            yield send_event("error", {"message": "Need at least 2 keywords to cluster"})
            return

        total_keywords = len(keywords)

        # Step 1: Preprocessing
        yield send_event("step", {
            "step": "preprocess",
            "status": "processing",
            "message": f"Processing {total_keywords} keywords..."
        })

        segmented = apply_vietnamese_word_segmentation(keywords, request.language)

        yield send_event("step", {
            "step": "preprocess",
            "status": "completed",
            "message": f"Preprocessed {total_keywords} keywords"
        })

        # Step 2: Embeddings
        yield send_event("step", {
            "step": "embed",
            "status": "processing",
            "message": "Generating embeddings with OpenAI..."
        })

        try:
            embeddings = get_openai_embeddings_parallel(segmented, api_key)
        except Exception as e:
            yield send_event("error", {"message": f"OpenAI API error: {str(e)}"})
            return

        yield send_event("step", {
            "step": "embed",
            "status": "completed",
            "message": f"Generated {total_keywords} embeddings (3,072 dimensions)"
        })

        # Step 3: Clustering
        yield send_event("step", {
            "step": "cluster",
            "status": "processing",
            "message": "Running HDBSCAN clustering..."
        })

        # Use provided config or default to recommended
        clustering_config = request.clustering_config or CLUSTERING_PRESETS["recommended"]
        clusters = cluster_with_hdbscan(embeddings, request.clustering_blocks, config=clustering_config)
        num_clusters = len(np.unique(clusters))

        yield send_event("step", {
            "step": "cluster",
            "status": "completed",
            "message": f"Found {num_clusters} clusters"
        })

        # Step 4: Labeling
        yield send_event("step", {
            "step": "label",
            "status": "processing",
            "message": f"Generating labels with GPT-4o-mini..."
        })

        try:
            cluster_labels = generate_cluster_labels_fast(
                keywords, clusters, api_key, request.language
            )
        except Exception:
            cluster_labels = {int(c): f"Cluster {c}" for c in np.unique(clusters)}

        yield send_event("step", {
            "step": "label",
            "status": "completed",
            "message": f"Generated {num_clusters} cluster labels"
        })

        # Step 5: Visualization
        yield send_event("step", {
            "step": "visualize",
            "status": "processing",
            "message": "Creating UMAP visualization..."
        })

        embeddings_2d = create_2d_embeddings(embeddings, clusters)
        embeddings_3d = create_3d_embeddings(embeddings, clusters)

        yield send_event("step", {
            "step": "visualize",
            "status": "completed",
            "message": "Visualization ready"
        })

        # Final result
        yield send_event("complete", {
            "result": {
                "keywords": keywords,
                "segmented": segmented,
                "clusters": clusters.tolist(),
                "cluster_labels": cluster_labels,
                "embeddings_2d": embeddings_2d.tolist(),
                "embeddings_3d": embeddings_3d.tolist()
            }
        })

    except Exception as e:
        yield send_event("error", {"message": str(e)})


@app.get("/")
async def root():
    return {"message": "Keyword Clustering API", "status": "running", "version": "1.2.0"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vietnamese_nlp": VIETNAMESE_AVAILABLE,
        "features": ["streaming_progress", "parallel_embeddings", "gpt4o_mini_labeling"]
    }


@app.get("/api-status")
async def api_status():
    """Check if API key is configured in .env"""
    try:
        env_key = os.getenv("OPENAI_API_KEY")
        return {
            "configured": bool(env_key and env_key.strip()),
            "source": "environment" if env_key else None
        }
    except Exception as e:
        return {"configured": False, "error": str(e)}


@app.post("/cluster/stream")
async def cluster_keywords_stream(request: ClusterRequest):
    """Streaming endpoint with real-time progress updates"""
    return StreamingResponse(
        cluster_with_progress(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/cluster", response_model=ClusterResponse)
async def cluster_keywords(request: ClusterRequest):
    """Non-streaming endpoint for backwards compatibility"""
    try:
        api_key = get_api_key(request.api_key)
        keywords = [kw.strip() for kw in request.keywords if kw.strip()]
        keywords = list(dict.fromkeys(keywords))

        if len(keywords) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 keywords")

        segmented = apply_vietnamese_word_segmentation(keywords, request.language)
        embeddings = get_openai_embeddings_parallel(segmented, api_key)
        clustering_config = request.clustering_config or CLUSTERING_PRESETS["recommended"]
        clusters = cluster_with_hdbscan(embeddings, request.clustering_blocks, config=clustering_config)

        try:
            cluster_labels = generate_cluster_labels_fast(keywords, clusters, api_key, request.language)
        except Exception:
            cluster_labels = {int(c): f"Cluster {c}" for c in np.unique(clusters)}

        embeddings_2d = create_2d_embeddings(embeddings, clusters)
        embeddings_3d = create_3d_embeddings(embeddings, clusters)

        return ClusterResponse(
            keywords=keywords,
            segmented=segmented,
            clusters=clusters.tolist(),
            cluster_labels=cluster_labels,
            embeddings_2d=embeddings_2d.tolist(),
            embeddings_3d=embeddings_3d.tolist()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clustering-config")
async def get_clustering_config():
    """
    Get available clustering configuration options and presets.
    Returns user-friendly descriptions for the frontend.
    """
    return {
        "parameters": {
            "granularity": {
                "name": "Cluster Granularity",
                "description": "How fine-grained should the clusters be? Lower = fewer large clusters, Higher = many smaller clusters.",
                "min": 1,
                "max": 10,
                "default": 5,
                "low_label": "Few large clusters",
                "high_label": "Many small clusters"
            },
            "min_keywords_per_cluster": {
                "name": "Minimum Keywords per Cluster",
                "description": "The minimum number of keywords required to form a cluster. Smaller values allow more niche clusters.",
                "min": 3,
                "max": 30,
                "default": 8,
                "low_label": "Allow small clusters (3+)",
                "high_label": "Require larger clusters (30+)"
            },
            "cluster_coherence": {
                "name": "Cluster Coherence",
                "description": "How similar must keywords be to belong to the same cluster? Higher = stricter grouping, more outliers.",
                "min": 1,
                "max": 10,
                "default": 5,
                "low_label": "Loose (include more)",
                "high_label": "Strict (only similar)"
            }
        },
        "presets": {
            "recommended": {
                "name": "Recommended",
                "description": "Balanced settings that work well for most keyword sets",
                "config": CLUSTERING_PRESETS["recommended"].model_dump()
            },
            "few_large": {
                "name": "Few Large Clusters",
                "description": "Create fewer, broader clusters with more keywords each",
                "config": CLUSTERING_PRESETS["few_large"].model_dump()
            },
            "many_small": {
                "name": "Many Small Clusters",
                "description": "Create more granular clusters with smaller, focused groups",
                "config": CLUSTERING_PRESETS["many_small"].model_dump()
            },
            "strict_quality": {
                "name": "Strict Quality",
                "description": "Prioritize cluster coherence - only group very similar keywords",
                "config": CLUSTERING_PRESETS["strict_quality"].model_dump()
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
