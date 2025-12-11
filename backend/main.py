"""
FastAPI Backend for Keyword Clustering
Uses the same Python dependencies as the Streamlit app
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
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
    description="AI-Powered Keyword Clustering with OpenAI & GPT-4o",
    version="1.0.0"
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ClusterRequest(BaseModel):
    keywords: List[str]
    api_key: str
    language: str = "Vietnamese"
    clustering_blocks: int = 1000


class ClusterResponse(BaseModel):
    keywords: List[str]
    segmented: List[str]
    clusters: List[int]
    cluster_labels: Dict[int, str]
    embeddings_2d: List[List[float]]


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


def get_openai_embeddings(keywords: List[str], api_key: str, model: str = "text-embedding-3-large") -> np.ndarray:
    """Get embeddings from OpenAI API"""
    client = OpenAI(api_key=api_key)

    batch_size = 100
    all_embeddings = []

    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch,
            encoding_format="float"
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

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


def cluster_with_hdbscan(embeddings: np.ndarray, n_blocks: int) -> np.ndarray:
    """Cluster embeddings using HDBSCAN in blocks with outlier reassignment"""
    hdbscan_params = {
        'min_cluster_size': 2,
        'min_samples': 1,
        'cluster_selection_epsilon': 0.1,
        'cluster_selection_method': 'leaf',
        'core_dist_n_jobs': -1
    }

    min_block_size = 10
    if len(embeddings) < min_block_size * 2:
        clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        all_clusters = clusterer.fit_predict(embeddings)
        if np.sum(all_clusters == -1) > 0:
            all_clusters = reassign_outliers_to_nearest_cluster(embeddings, all_clusters)
        return all_clusters

    max_blocks = len(embeddings) // min_block_size
    n_blocks = min(n_blocks, max_blocks)

    block_size = len(embeddings) // n_blocks
    all_clusters = np.full(len(embeddings), -1, dtype=int)

    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else len(embeddings)
        block = embeddings[start:end]

        clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        block_clusters = clusterer.fit_predict(block)
        all_clusters[start:end] = block_clusters

    if np.sum(all_clusters == -1) > 0:
        all_clusters = reassign_outliers_to_nearest_cluster(embeddings, all_clusters)

    return all_clusters


def generate_cluster_labels_with_gpt4(
    keywords: List[str],
    clusters: np.ndarray,
    api_key: str,
    language: str = "Vietnamese"
) -> Dict[int, str]:
    """Generate high-quality cluster labels using GPT-4o"""
    client = OpenAI(api_key=api_key)
    unique_clusters = np.unique(clusters)
    cluster_labels = {}

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_labels[int(cluster_id)] = "Outliers / Uncategorized"
            continue

        # Get sample keywords from this cluster
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
3. Use the most representative keywords
4. Respond with ONLY the label, no explanation

Label:"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )

        label = response.choices[0].message.content.strip()
        cluster_labels[int(cluster_id)] = label

    return cluster_labels


def create_2d_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP"""
    umap_model = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    return umap_model.fit_transform(embeddings)


@app.get("/")
async def root():
    return {"message": "Keyword Clustering API", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vietnamese_nlp": VIETNAMESE_AVAILABLE
    }


@app.post("/cluster", response_model=ClusterResponse)
async def cluster_keywords(request: ClusterRequest):
    """Main clustering endpoint"""
    try:
        keywords = request.keywords

        # Remove empty and duplicate keywords
        keywords = [kw.strip() for kw in keywords if kw.strip()]
        keywords = list(dict.fromkeys(keywords))  # Preserve order while removing duplicates

        if len(keywords) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 keywords to cluster")

        # Step 1: Vietnamese word segmentation
        segmented = apply_vietnamese_word_segmentation(keywords, request.language)

        # Step 2: Get OpenAI embeddings
        try:
            embeddings = get_openai_embeddings(segmented, request.api_key)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"OpenAI API error: {str(e)}")

        # Step 3: Cluster with HDBSCAN
        clusters = cluster_with_hdbscan(embeddings, request.clustering_blocks)

        # Step 4: Generate cluster labels with GPT-4o
        try:
            cluster_labels = generate_cluster_labels_with_gpt4(
                keywords, clusters, request.api_key, request.language
            )
        except Exception as e:
            # Fallback to simple labels if GPT-4o fails
            cluster_labels = {int(c): f"Cluster {c}" for c in np.unique(clusters)}

        # Step 5: Create 2D embeddings for visualization
        embeddings_2d = create_2d_embeddings(embeddings)

        return ClusterResponse(
            keywords=keywords,
            segmented=segmented,
            clusters=clusters.tolist(),
            cluster_labels=cluster_labels,
            embeddings_2d=embeddings_2d.tolist()
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
