# Keyword Clustering

AI-powered semantic keyword clustering using OpenAI embeddings and HDBSCAN. Built with Next.js and FastAPI.

## Features

- **3,072-dimensional embeddings** via OpenAI `text-embedding-3-large`
- **Density-based clustering** with HDBSCAN (no predefined cluster count)
- **AI-generated labels** using GPT-4o-mini for human-readable cluster names
- **Interactive visualizations** with 2D/3D UMAP projections
- **Real-time progress** via Server-Sent Events (SSE)
- **Vietnamese support** with underthesea word segmentation
- **Zero outliers** via automatic reassignment to nearest clusters

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Next.js 15, React 19, TypeScript, Tailwind CSS |
| Backend | FastAPI, Python 3.12, Uvicorn |
| Embeddings | OpenAI `text-embedding-3-large` (3,072 dimensions) |
| Clustering | HDBSCAN with UMAP dimensionality reduction |
| Labeling | GPT-4o-mini |
| Visualization | Plotly.js, UMAP |

## Architecture

```
┌─────────────────┐     SSE Stream      ┌─────────────────┐
│   Next.js App   │ ◄─────────────────► │  FastAPI Server │
│   (Port 3000)   │                     │   (Port 8000)   │
└────────┬────────┘                     └────────┬────────┘
         │                                       │
         │ Upload Excel                          │ OpenAI API
         │ View Results                          │ HDBSCAN
         │ Export Excel                          │ UMAP
         │                                       │ GPT-4o-mini
```

## Quick Start

### Prerequisites

- Node.js 20+
- Python 3.12+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Local Development

1. **Clone and setup environment**
   ```bash
   git clone https://github.com/yourusername/keyword-clustering.git
   cd keyword-clustering
   cp .env.example .env
   # Add your OPENAI_API_KEY to .env
   ```

2. **Start the backend**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn main:app --reload --port 8000
   ```

3. **Start the frontend** (new terminal)
   ```bash
   cd nextjs-app
   npm install
   npm run dev
   ```

4. **Open** http://localhost:3000

## Clustering Algorithm

### HDBSCAN Configuration

The clustering uses a two-stage approach for high-dimensional embeddings:

**Stage 1: UMAP Dimensionality Reduction**
```python
UMAP(
    n_components=50,        # Reduce 3,072 dims → 50 dims
    n_neighbors=15,         # Local neighborhood size
    min_dist=0.0,           # Allow tight clusters
    metric='cosine',        # Semantic similarity metric
)
```

**Stage 2: HDBSCAN Clustering**
```python
HDBSCAN(
    min_cluster_size=max(5, n_samples // 200),  # 0.5% of dataset, capped at 30
    min_samples=3,                               # Core point threshold
    cluster_selection_epsilon=0.0,               # No distance threshold
    cluster_selection_method='eom',              # Excess of Mass
    metric='euclidean',                          # After UMAP reduction
)
```

### Post-Processing

1. **Outlier Reassignment**: Points labeled as outliers (-1) are reassigned to their nearest cluster using cosine similarity
2. **Cluster Merging**: If clusters exceed 20, similar clusters (cosine similarity > 0.90) are merged

### Why These Parameters?

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `min_cluster_size` | 0.5% of data (max 30) | Prevents tiny clusters while allowing dataset-appropriate sizing |
| `min_samples=3` | Conservative | Requires 3 neighbors to form core points, reducing noise sensitivity |
| `cluster_selection_method='eom'` | Excess of Mass | Better at finding clusters of varying densities |
| `UMAP n_components=50` | 50 dims | Balances information preservation with computational efficiency |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and version |
| `GET` | `/health` | Health check with feature flags |
| `GET` | `/api-status` | Check if OpenAI key is configured |
| `POST` | `/cluster/stream` | SSE streaming clustering (recommended) |
| `POST` | `/cluster` | Synchronous clustering |

### Request Body
```json
{
  "keywords": ["keyword 1", "keyword 2", "..."],
  "language": "Vietnamese",
  "clustering_blocks": 1000,
  "api_key": "sk-..."  // Optional if set in .env
}
```

### SSE Events
```
data: {"type": "step", "step": "preprocess", "status": "processing", ...}
data: {"type": "step", "step": "embed", "status": "completed", ...}
data: {"type": "complete", "result": {...}}
```

## Deploy to Railway

### One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

### Manual Setup

1. **Create a new Railway project**

2. **Add Backend Service**
   - New Service → GitHub Repo → Select this repo
   - Set root directory: `backend`
   - Add environment variable: `OPENAI_API_KEY`

3. **Add Frontend Service**
   - New Service → GitHub Repo → Select this repo
   - Set root directory: `nextjs-app`
   - Add environment variable: `BACKEND_URL` = `https://your-backend.railway.app`

4. **Generate domains** for both services

### Environment Variables

| Variable | Service | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Backend | Your OpenAI API key |
| `BACKEND_URL` | Frontend | Backend service URL (Railway internal or public) |

## Cost Estimation

| Component | Model | Cost per 1K keywords |
|-----------|-------|---------------------|
| Embeddings | text-embedding-3-large | ~$0.13 |
| Labels | GPT-4o-mini | ~$0.01-0.05 |
| **Total** | | **~$0.15-0.20** |

*Actual costs depend on keyword length and number of clusters.*

## Project Structure

```
keyword-clustering/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # Docker build config
│   └── railway.toml         # Railway deployment config
├── nextjs-app/
│   ├── src/
│   │   ├── app/             # Next.js App Router
│   │   └── components/      # React components
│   ├── Dockerfile           # Multi-stage Docker build
│   └── railway.toml         # Railway deployment config
├── .env.example             # Environment template
└── README.md
```

## Development

### Backend Development
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd nextjs-app
npm run dev
```

### Run Tests
```bash
# Backend
cd backend && pytest

# Frontend
cd nextjs-app && npm test
```

## License

MIT

---

**Built with Next.js + FastAPI | Powered by OpenAI**
