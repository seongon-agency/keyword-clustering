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


class ClusterRequest(BaseModel):
    keywords: List[str]
    api_key: Optional[str] = None
    language: str = "Vietnamese"
    clustering_blocks: int = 1000


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


def reassign_outliers_to_nearest_cluster(embeddings: np.ndarray, clusters: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Reassign outlier points to their nearest cluster using k-NN voting.
    Uses majority vote among k nearest neighbors for more robust assignment.
    """
    outlier_indices = np.where(clusters == -1)[0]
    non_outlier_indices = np.where(clusters != -1)[0]

    if len(outlier_indices) == 0 or len(non_outlier_indices) == 0:
        return clusters

    outlier_embeddings = embeddings[outlier_indices]
    non_outlier_embeddings = embeddings[non_outlier_indices]
    similarities = cosine_similarity(outlier_embeddings, non_outlier_embeddings)

    new_clusters = clusters.copy()
    k_actual = min(k, len(non_outlier_indices))  # Can't use more neighbors than we have

    for i, outlier_idx in enumerate(outlier_indices):
        # Get top-k nearest neighbors
        top_k_indices = np.argsort(similarities[i])[-k_actual:]
        top_k_clusters = [clusters[non_outlier_indices[idx]] for idx in top_k_indices]

        # Majority vote among k nearest neighbors
        cluster_votes = Counter(top_k_clusters)
        best_cluster = cluster_votes.most_common(1)[0][0]
        new_clusters[outlier_idx] = best_cluster

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


def subdivide_cluster(embeddings: np.ndarray, min_cluster_size: int = 10) -> np.ndarray:
    """
    Attempt to subdivide a single cluster into smaller sub-clusters.
    Returns cluster labels (0, 1, 2, ...) or all zeros if can't split.
    """
    n_samples = len(embeddings)
    if n_samples < min_cluster_size * 2:
        return np.zeros(n_samples, dtype=int)

    # Use smaller n_neighbors for subdivision to find finer structure
    n_components = min(30, n_samples - 1)
    n_neighbors = min(8, n_samples - 1)

    try:
        umap_reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric='euclidean',  # Already reduced, use euclidean
            random_state=42,
            n_jobs=1
        )
        reduced = umap_reducer.fit_transform(embeddings)

        sub_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom',
            metric='euclidean'
        )
        sub_labels = sub_clusterer.fit_predict(reduced)

        # Check if we actually got multiple clusters (not just one + outliers)
        unique_labels = set(sub_labels) - {-1}
        if len(unique_labels) >= 2:
            # Reassign outliers
            if np.sum(sub_labels == -1) > 0:
                sub_labels = reassign_outliers_to_nearest_cluster(reduced, sub_labels)
            return sub_labels
    except Exception as e:
        print(f"[DEBUG] Subdivision failed: {e}")

    return np.zeros(n_samples, dtype=int)


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


def cluster_with_hdbscan(embeddings: np.ndarray, n_blocks: int) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN with UMAP dimensionality reduction.
    Recursively subdivides oversized clusters for balanced results.
    """
    n_samples = len(embeddings)

    # Tighter thresholds for better balance
    max_cluster_size = max(250, n_samples // 12)  # ~8% of dataset or 250
    min_cluster_size = max(5, n_samples // 500)  # 0.2% of dataset
    min_cluster_size = min(min_cluster_size, 15)  # Cap at 15
    tiny_cluster_threshold = max(3, min_cluster_size // 2)  # Clusters smaller than this get merged

    # Reduce dimensions with UMAP first
    n_components = min(50, n_samples - 1)
    umap_reducer = UMAP(
        n_components=n_components,
        n_neighbors=10,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        n_jobs=1
    )
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    hdbscan_params = {
        'min_cluster_size': min_cluster_size,
        'min_samples': 2,
        'cluster_selection_epsilon': 0.0,
        'cluster_selection_method': 'eom',
        'metric': 'euclidean',
        'core_dist_n_jobs': -1
    }

    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    all_clusters = clusterer.fit_predict(reduced_embeddings)

    print(f"[DEBUG] HDBSCAN initial: {len(np.unique(all_clusters))} clusters, outliers: {np.sum(all_clusters == -1)}/{n_samples}")

    # Reassign outliers using k-NN voting
    if np.sum(all_clusters == -1) > 0:
        all_clusters = reassign_outliers_to_nearest_cluster(reduced_embeddings, all_clusters, k=5)

    # Recursively subdivide oversized clusters
    max_iterations = 5
    for iteration in range(max_iterations):
        cluster_sizes = {}
        for c in np.unique(all_clusters):
            cluster_sizes[c] = np.sum(all_clusters == c)

        oversized = [c for c, size in cluster_sizes.items() if size > max_cluster_size]
        if not oversized:
            break

        print(f"[DEBUG] Iteration {iteration + 1}: Subdividing {len(oversized)} oversized clusters")

        next_cluster_id = max(all_clusters) + 1
        new_clusters = all_clusters.copy()

        for cluster_id in oversized:
            cluster_mask = all_clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = reduced_embeddings[cluster_indices]

            sub_labels = subdivide_cluster(cluster_embeddings, min_cluster_size=min_cluster_size)

            unique_sub = set(sub_labels)
            if len(unique_sub) >= 2:
                for sub_id in unique_sub:
                    if sub_id == 0:
                        continue
                    sub_mask = sub_labels == sub_id
                    global_indices = cluster_indices[sub_mask]
                    new_clusters[global_indices] = next_cluster_id
                    next_cluster_id += 1

                print(f"[DEBUG] Cluster {cluster_id} ({len(cluster_indices)} pts) -> {len(unique_sub)} sub-clusters")

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

        clusters = cluster_with_hdbscan(embeddings, request.clustering_blocks)
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
        clusters = cluster_with_hdbscan(embeddings, request.clustering_blocks)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
