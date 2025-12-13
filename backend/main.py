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


def reassign_outliers_to_nearest_cluster(embeddings: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Reassign outlier points to their nearest cluster"""
    outlier_indices = np.where(clusters == -1)[0]
    non_outlier_indices = np.where(clusters != -1)[0]

    if len(outlier_indices) == 0 or len(non_outlier_indices) == 0:
        return clusters

    outlier_embeddings = embeddings[outlier_indices]
    non_outlier_embeddings = embeddings[non_outlier_indices]
    similarities = cosine_similarity(outlier_embeddings, non_outlier_embeddings)

    new_clusters = clusters.copy()
    for i, outlier_idx in enumerate(outlier_indices):
        nearest_non_outlier_idx = non_outlier_indices[np.argmax(similarities[i])]
        new_clusters[outlier_idx] = clusters[nearest_non_outlier_idx]

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


def cluster_with_hdbscan(embeddings: np.ndarray, n_blocks: int) -> np.ndarray:
    """
    Cluster embeddings using HDBSCAN with UMAP dimensionality reduction.
    """
    n_samples = len(embeddings)

    # Reduce dimensions with UMAP first (critical for high-dim embeddings)
    # This avoids the curse of dimensionality with 1536-dim OpenAI embeddings
    n_components = min(50, n_samples - 1)  # UMAP target dimensions
    umap_reducer = UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        n_jobs=1
    )
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    # Conservative min_cluster_size
    min_cluster_size = max(5, n_samples // 200)  # 0.5% of dataset
    min_cluster_size = min(min_cluster_size, 30)  # Cap at 30

    hdbscan_params = {
        'min_cluster_size': min_cluster_size,
        'min_samples': 3,
        'cluster_selection_epsilon': 0.0,
        'cluster_selection_method': 'eom',
        'metric': 'euclidean',
        'core_dist_n_jobs': -1
    }

    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    all_clusters = clusterer.fit_predict(reduced_embeddings)

    print(f"[DEBUG] HDBSCAN found {len(np.unique(all_clusters))} clusters, outliers: {np.sum(all_clusters == -1)}/{n_samples}")

    # Reassign outliers to nearest cluster (using reduced embeddings for consistency)
    if np.sum(all_clusters == -1) > 0:
        all_clusters = reassign_outliers_to_nearest_cluster(reduced_embeddings, all_clusters)

    num_clusters = len(np.unique(all_clusters))
    print(f"[DEBUG] Final: {num_clusters} clusters")

    # Light merge only if too many clusters
    if num_clusters > 20:
        all_clusters = merge_similar_clusters(reduced_embeddings, all_clusters, similarity_threshold=0.90)

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
